"""Run: python validation/evaluate.py [--rewrite-mode model]."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pymupdf

from src.config import TOP_K, USER_ACCESS, CHUNK_OVERLAP, CHUNK_SIZE, EMBEDDING_MODEL

OUTPUT = ROOT / 'validation/results.json'


def check_dataset():
    permission_warnings = []
    dataset = json.loads((ROOT / "validation/validation_data.json").read_text(encoding="utf-8"))
    cases = dataset["cases"]
    fingerprint = dataset["dataset_fingerprint"]
    assert fingerprint["embedding_model"] == EMBEDDING_MODEL
    assert fingerprint["chunk_size"] == CHUNK_SIZE
    assert fingerprint["chunk_overlap"] == CHUNK_OVERLAP
    assert len(cases) == 100
    assert len({c["id"] for c in cases}) == 100
    assert len({c["session_id"] for c in cases}) == 100
    companies = {company for allowed in USER_ACCESS.values() for company in allowed}
    assert {c["company"] for c in cases} == companies
    page_text = {}
    for company in sorted(companies):
        path = ROOT / "data" / f"{company}.pdf"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == fingerprint["pdf_sha256"][company], path
        with pymupdf.open(path) as pdf:
            for number, page in enumerate(pdf, 1):
                page_text[(path.name, number)] = " ".join(page.get_text(sort=True).split())
        subset = [c for c in cases if c["company"] == company]
        assert Counter(c["difficulty"] for c in subset) == {"easy": 10, "medium": 5, "hard": 5}
        assert Counter(c["kind"] for c in subset) == {"answerable": 15, "follow_up": 2, "unanswerable": 2, "denied": 1}

    for case in cases:
        label = case["id"]
        assert case["question"] and case["expected_answer"] and case["required_answer_checks"], label
        allowed = case["company"] in USER_ACCESS.get(case["email"], [])
        if allowed != (case["kind"] != "denied"):
            permission_warnings.append(
                f"{label}: current permissions for {case['email']} differ from the dataset's expected access."
            )
        positive = case["kind"] in {"answerable", "follow_up"}
        assert case["expected_behavior"] == ("answer" if positive else case["kind"]), label
        assert bool(case["relevant_pages"]) == positive, label
        assert bool(case["reference_evidence"]) == positive, label
        assert bool(case["history"]) == (case["kind"] == "follow_up"), label
        if case["kind"] == "follow_up":
            assert case["expected_standalone_question"], label
            assert [m["role"] for m in case["history"]] == ["user", "assistant"], label
        references = {(r["source"], r["page"]) for r in case["relevant_pages"]}
        assert references == {(e["source"], e["page"]) for e in case["reference_evidence"]}, label
        for evidence in case["reference_evidence"]:
            assert evidence["source"] == f"{case['company']}.pdf", label
            page = page_text[(evidence["source"], evidence["page"])]
            assert evidence["anchor"] in page, (label, evidence["anchor"])
            assert evidence["excerpt"] in page, label
    print("PASS: 100 cases; each company has 10 easy, 5 medium, 5 hard.")
    print("PASS: PDF hashes, evidence excerpts/pages and follow-up structure.")
    print("85 answerable/follow-up cases; 10 unanswerable; 5 denied.")
    print("These checks do not measure retrieval or answer accuracy.")
    if permission_warnings:
        print(f"WARNING: {len(permission_warnings)} cases have permissions inconsistent with current config.")
    return permission_warnings


def summarize(rows):
    total = len(rows)
    found = sum(row['found_relevant_page'] for row in rows)
    return {
        'questions_tested': total,
        'questions_with_relevant_page': found,
        'retrieval_success_percent': round(100 * found / total, 2) if total else None,
        'first_result_correct_percent': round(100 * sum(r['first_result_correct'] for r in rows) / total, 2) if total else None,
        'errors': sum('error' in row for row in rows),
        'average_search_seconds': round(sum(r['seconds'] for r in rows) / total, 3) if total else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rewrite-mode', choices=('gold', 'model'), default='gold',
                        help='gold uses labelled follow-up questions; model uses the LLM to rewrite them')
    args = parser.parse_args()
    started = time.perf_counter()
    path = ROOT / 'validation/validation_data.json'
    report = {
        'created_at': datetime.now(timezone.utc).isoformat(),
        'status': 'error',
        'test_type': 'Document retrieval and explicit company-access checks',
        'rewrite_mode': args.rewrite_mode,
        'dataset_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'current_user_access': USER_ACCESS,
        'meaning': f'Success means at least one labelled relevant PDF page appeared in the top {TOP_K} chunks. Errors count as misses.',
        'limitations': [
            'This does not measure LLM answer accuracy, citation correctness, or UI response time.',
            'Unanswerable questions are not scored; they need answer-generation checks.',
            'Gold mode uses labelled follow-up rewrites, so it does not test LLM rewriting.',
            'Page labels are curated, not exhaustive. This is a development dataset.',
            'Access checks test explicit company requests, not all possible attempts to bypass permissions.',
        ],
    }
    try:
        report['permission_warnings'] = check_dataset()
        dataset = json.loads(path.read_text(encoding='utf-8'))
        report['dataset_questions'] = len(dataset['cases'])
        from src import rag
        model, collection = rag.load_retriever()
        if collection.count() != dataset['dataset_fingerprint']['chunk_count']:
            raise ValueError('Index chunk count differs from dataset. Rebuild/review labels.')
        rows, access = [], []
        for case in dataset['cases']:
            if case['kind'] == 'denied':
                item = {'id': case['id'], 'question': case['question'], 'passed': False}
                try:
                    rag.check_access(case['question'], case['email'])
                    item['detail'] = 'Request was allowed, but the dataset expects denial.'
                except PermissionError:
                    item.update(passed=True, detail='Unauthorized company request was blocked.')
                except Exception as error:
                    item['detail'] = f'{type(error).__name__}: {error}'
                access.append(item)
                continue
            if case['kind'] not in {'answerable', 'follow_up'}:
                continue
            row = {key: case[key] for key in ('id', 'company', 'difficulty', 'question')}
            row.update(found_relevant_page=False, first_result_correct=False)
            timer = time.perf_counter()
            try:
                rag.check_access(case['question'], case['email'])
                query = (case.get('expected_standalone_question', case['question'])
                         if args.rewrite_mode == 'gold'
                         else rag.rewrite_question(case['question'], case['history']))
                result = rag.retrieve(query, case['email'], model, collection)
                metadata = result['metadatas'][0]
                if any(m['company'] not in USER_ACCESS[case['email']] for m in metadata):
                    raise RuntimeError('Retrieved an unauthorized company.')
                pages = list(dict.fromkeys((m['source'], m['page']) for m in metadata))
                expected = {(r['source'], r['page']) for r in case['relevant_pages']}
                row.update(retrieval_query=query,
                           found_relevant_page=bool(set(pages) & expected),
                           first_result_correct=bool(pages and pages[0] in expected),
                           retrieved_pages=[{'source': source, 'page': page} for source, page in pages])
            except Exception as error:
                row['error'] = f'{type(error).__name__}: {error}'
            row['seconds'] = round(time.perf_counter() - timer, 4)
            rows.append(row)
        report['retrieval'] = summarize(rows)
        report['summary'] = (f"Found a relevant page for {report['retrieval']['questions_with_relevant_page']} "
                             f"of {len(rows)} questions ({report['retrieval']['retrieval_success_percent']}%).")
        report['by_company'] = {c: summarize([r for r in rows if r['company'] == c]) for c in sorted({r['company'] for r in rows})}
        report['by_difficulty'] = {d: summarize([r for r in rows if r['difficulty'] == d]) for d in ('easy', 'medium', 'hard')}
        report['access_checks'] = {'tested': len(access), 'passed': sum(r['passed'] for r in access), 'cases': access}
        report['not_scored'] = {'unanswerable_questions': Counter(c['kind'] for c in dataset['cases'])['unanswerable']}
        report['cases'] = rows
        report['status'] = ('completed_with_issues' if report['permission_warnings'] or report['retrieval']['errors'] or not all(r['passed'] for r in access) else 'completed')
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
    finally:
        if args.rewrite_mode == 'model':
            from src.inference import stop_inference
            stop_inference()
        report['total_seconds_including_setup'] = round(time.perf_counter() - started, 2)
        OUTPUT.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(report.get('summary', report.get('error')))
    print('Status:', report['status'])
    print('Saved:', OUTPUT)
    if report['status'] != 'completed':
        raise SystemExit('Review issues in validation/results.json.')


if __name__ == '__main__':
    main()
