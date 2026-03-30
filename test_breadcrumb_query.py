"""Quick test: scan + AND filter + Python sort (the new approach)."""
import lancedb

db = lancedb.connect(r"C:\Users\jensenl\AppData\Roaming\as-help-mcp\data\as6\.ashelp_lance")
table = db.open_table("help_pages")

TARGET = "d23adc69-c0f8-4255-ab80-a3c08e91e62e"


def test_breadcrumb_scan(query_label, terms):
    """Scan with AND breadcrumb filter, sort by match count in Python."""
    print(f"\n=== {query_label} (terms: {terms}) ===")
    bc_conditions = [f"lower(breadcrumb_path) LIKE '%{t}%'" for t in terms]
    bc_filter = " AND ".join(bc_conditions)
    print(f"  Filter: {bc_filter}")

    try:
        results = table.search().where(bc_filter).limit(200).to_list()
        print(f"  Raw scan: {len(results)} results")

        # Sort by match count, then by breadcrumb length (shorter=better)
        def sort_key(row):
            bc = (row.get("breadcrumb_path") or "").lower()
            hits = sum(1 for t in terms if t in bc)
            return (-hits, len(bc))

        results.sort(key=sort_key)
        found = False
        for i, r in enumerate(results[:15]):
            bc = (r.get("breadcrumb_path") or "").lower()
            hits = sum(1 for t in terms if t in bc)
            marker = " <<<" if r['page_id'] == TARGET else ""
            print(f"  #{i+1} hits={hits} | {r['page_id'][:8]}... | {r['title'][:50]} | bc: {r['breadcrumb_path'][:70]}{marker}")
            if r['page_id'] == TARGET:
                found = True
        if not found:
            for i, r in enumerate(results):
                if r['page_id'] == TARGET:
                    print(f"  >>> Target found at rank #{i+1} (of {len(results)})")
                    found = True
                    break
            if not found:
                print(f"  >>> Target NOT in results at all!")
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")


# Test queries
test_breadcrumb_scan("ACP10 revision information", ["acp10", "revision", "information"])
test_breadcrumb_scan("ACP10 changelog", ["acp10"])  # "changelog" won't be in breadcrumb
test_breadcrumb_scan("ACP10 version history", ["acp10", "version"])
test_breadcrumb_scan("ACP10 revision", ["acp10", "revision"])
