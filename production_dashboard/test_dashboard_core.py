import unittest
from pathlib import Path

from dashboard_core import (PRODUCTION_COLUMNS, read_csv, csv_bytes, validate_production,
                            merge_lines, equipment_summary, pareto_summary, daily_summary, join_quality)


def row(lot, day="2026-10-01", inspected=100, good=90, kind="A"):
    return dict(lot_id=lot, work_date=day, equipment_id="EQ", inspected_qty=str(inspected),
                good_qty=str(good), defect_type=kind)


class DashboardChecks(unittest.TestCase):
    def test_actual_totals_and_join(self):
        root = Path(__file__).parent
        columns, source = read_csv((root/"production.csv").read_bytes())
        valid, rejected = validate_production(columns, source)
        self.assertEqual(len(valid), 36)
        self.assertFalse(rejected)
        daily, cards = daily_summary(valid)
        self.assertEqual(sum(r["inspected_qty"] for r in daily), 38950)
        self.assertEqual(sum(r["good_qty"] for r in daily), 36930)
        self.assertEqual(len(cards), 6)
        self.assertEqual([r["defect_qty"] for r in pareto_summary(valid)], [789, 616, 615])
        qa_columns, qa = read_csv((root/"quality.csv").read_bytes())
        joined = join_quality(valid, qa_columns, qa)
        self.assertEqual(len(joined), len(qa))
        self.assertTrue(all(r["production_matched"] for r in joined))

    def test_weighted_yield_not_average(self):
        valid, _ = validate_production(PRODUCTION_COLUMNS,[row("A",inspected=10,good=5),row("B",inspected=90,good=90)])
        self.assertEqual(equipment_summary(valid)[0]["yield_pct"],95)
        self.assertEqual(daily_summary(valid)[0][0]["yield_pct"],95)

    def test_gap_sort_zero_and_segments(self):
        valid, _ = validate_production(PRODUCTION_COLUMNS,[row("B","2026-10-03",100,0),row("A")])
        daily, _ = daily_summary(valid)
        self.assertEqual([r["work_date"] for r in daily],["2026-10-01","2026-10-02","2026-10-03"])
        self.assertIsNone(daily[1]["yield_pct"])
        self.assertIsNone(daily[1]["inspected_qty"])
        self.assertEqual(daily[2]["yield_pct"],0)
        self.assertNotEqual(daily[0]["segment"],daily[2]["segment"])

    def test_reject_entire_duplicate_group_and_bad_values(self):
        source = [row("A"),row(" A "),row("B","2026-02-30"),row("C",good=101),row("D",inspected="1.2"),row("E")]
        valid, bad = validate_production(PRODUCTION_COLUMNS,source)
        self.assertEqual([r["lot_id"] for r in valid],["E"])
        self.assertEqual(len(bad),5)

    def test_missing_type_counted_and_zero_defects_undefined_share(self):
        valid, _ = validate_production(PRODUCTION_COLUMNS,[row("A",kind=" "),row("B",kind="B")])
        summary = pareto_summary(valid)
        self.assertEqual(sum(r["defect_qty"] for r in summary),20)
        self.assertIn("미분류",[r["defect_type"] for r in summary])
        valid, _ = validate_production(PRODUCTION_COLUMNS,[row("A",good=100)])
        self.assertIsNone(pareto_summary(valid)[0]["cumulative_pct"])

    def test_merge_deduplicates_full_rows_only(self):
        files = [("a.csv",csv_bytes([row("A")])),("b.csv",csv_bytes([row("A"),row("A",good=80)]))]
        self.assertEqual(len(merge_lines(files,True)),2)
        self.assertEqual(len(merge_lines(files,False)),3)
        with self.assertRaises(ValueError):
            merge_lines([files[0],("c.csv",b"other\n1\n")])

    def test_quality_keeps_all_inspections_and_unmatched(self):
        valid, _ = validate_production(PRODUCTION_COLUMNS,[row("A")])
        quality = [{"lot_id":"A","id":"1"},{"lot_id":"A","id":"2"},{"lot_id":"X","id":"3"}]
        joined = join_quality(valid,["lot_id","id"],quality)
        self.assertEqual(len(joined),3)
        self.assertEqual([r["production_matched"] for r in joined],[True,True,False])
        with self.assertRaises(ValueError):
            join_quality(valid+valid,["lot_id"],quality)

    def test_empty_selection_and_csv_contract(self):
        self.assertEqual(daily_summary([]),([],[]))
        self.assertEqual(equipment_summary([]),[])
        with self.assertRaises(ValueError):
            read_csv(b"a,a\n1,2\n")
        with self.assertRaises(ValueError):
            read_csv(b"a,b\n1\n")
        with self.assertRaises(ValueError):
            validate_production(["lot_id"],[{"lot_id":"A"}])


if __name__ == "__main__":
    unittest.main()
