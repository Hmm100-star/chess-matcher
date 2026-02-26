from __future__ import annotations

import io
import csv
import importlib
import os
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class RoundGenerationFlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmpdir = tempfile.TemporaryDirectory()
        db_path = Path(cls._tmpdir.name) / "test.db"
        os.environ["DATABASE_URL"] = f"sqlite:///{db_path}"
        os.environ["SECRET_KEY"] = "test-secret"

        cls.db = importlib.import_module("db")
        cls.app_module = importlib.import_module("app")

        cls.db.Base.metadata.drop_all(bind=cls.db.engine)
        cls.db.Base.metadata.create_all(bind=cls.db.engine)
        cls.app_module._tables_initialized = False

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmpdir.cleanup()

    def setUp(self) -> None:
        self.db.Base.metadata.drop_all(bind=self.db.engine)
        self.db.Base.metadata.create_all(bind=self.db.engine)
        self.app_module._tables_initialized = False
        self.client = self.app_module.app.test_client()

    def _extract_csrf(self, html: str) -> str:
        match = re.search(r'name="csrf_token" value="([^"]+)"', html)
        if not match:
            raise AssertionError("Missing CSRF token")
        return match.group(1)

    def _bootstrap_teacher_and_login(self) -> None:
        setup_page = self.client.get("/setup")
        setup_csrf = self._extract_csrf(setup_page.get_data(as_text=True))
        self.client.post(
            "/setup",
            data={
                "csrf_token": setup_csrf,
                "username": "teacher1",
                "password": "pw",
                "confirm_password": "pw",
            },
            follow_redirects=False,
        )

        login_page = self.client.get("/login")
        login_csrf = self._extract_csrf(login_page.get_data(as_text=True))
        self.client.post(
            "/login",
            data={"csrf_token": login_csrf, "username": "teacher1", "password": "pw"},
            follow_redirects=False,
        )

    def _create_classroom(self, name: str = "Class A") -> int:
        dashboard = self.client.get("/dashboard")
        csrf = self._extract_csrf(dashboard.get_data(as_text=True))
        response = self.client.post(
            "/dashboard",
            data={"csrf_token": csrf, "classroom_name": name},
            follow_redirects=True,
        )
        page = response.get_data(as_text=True)
        ids = re.findall(r"/classrooms/(\d+)", page)
        self.assertTrue(ids, "Expected classroom link in dashboard response")
        return int(ids[0])

    def _add_students(self, classroom_id: int, names: list[str]) -> None:
        for name in names:
            classroom_page = self.client.get(f"/classrooms/{classroom_id}")
            csrf = self._extract_csrf(classroom_page.get_data(as_text=True))
            self.client.post(
                f"/classrooms/{classroom_id}/students",
                data={"csrf_token": csrf, "student_name": name},
                follow_redirects=False,
            )

    def test_generate_matches_redirects_to_round_results_without_500(self) -> None:
        self._bootstrap_teacher_and_login()
        classroom_id = self._create_classroom()
        self._add_students(classroom_id, ["A", "B", "C", "D"])

        new_round_page = self.client.get(f"/classrooms/{classroom_id}/rounds/new")
        csrf = self._extract_csrf(new_round_page.get_data(as_text=True))
        response = self.client.post(
            f"/classrooms/{classroom_id}/rounds/new",
            data={"csrf_token": csrf, "win_weight": "0.7", "homework_weight": "0.3"},
            follow_redirects=True,
        )

        self.assertEqual(response.status_code, 200)
        body = response.get_data(as_text=True)
        self.assertIn("Round", body)
        self.assertIn("results", body)
        self.assertIn("white_entered_correct_", body)
        self.assertNotIn("We hit an unexpected error", body)

    def test_odd_player_count_creates_bye_and_renders_round(self) -> None:
        self._bootstrap_teacher_and_login()
        classroom_id = self._create_classroom("Class B")
        self._add_students(classroom_id, ["A", "B", "C"])

        new_round_page = self.client.get(f"/classrooms/{classroom_id}/rounds/new")
        csrf = self._extract_csrf(new_round_page.get_data(as_text=True))
        response = self.client.post(
            f"/classrooms/{classroom_id}/rounds/new",
            data={"csrf_token": csrf, "win_weight": "0.7", "homework_weight": "0.3"},
            follow_redirects=True,
        )

        self.assertEqual(response.status_code, 200)
        body = response.get_data(as_text=True)
        self.assertIn("Bye", body)
        self.assertNotIn("We hit an unexpected error", body)

    def test_import_then_generate_matches(self) -> None:
        self._bootstrap_teacher_and_login()
        classroom_id = self._create_classroom("Class Import")

        csv_data = "\n".join(
            [
                "Student Name,Total Wins,Total Losses,Total Ties,# Times Played White,# Times Played Black,Correct Homework,Incorrect Homework,Notes",
                "A,1,0,0,1,0,2,0,",
                "B,0,1,0,0,1,1,1,",
                "C,0,0,1,1,1,0,0,",
                "D,0,0,0,0,0,0,0,",
            ]
        )

        import_page = self.client.get(f"/classrooms/{classroom_id}/import")
        csrf = self._extract_csrf(import_page.get_data(as_text=True))
        self.client.post(
            f"/classrooms/{classroom_id}/import",
            data={
                "csrf_token": csrf,
                "replace_existing": "on",
                "file": (io.BytesIO(csv_data.encode("utf-8")), "students.csv"),
            },
            content_type="multipart/form-data",
            follow_redirects=False,
        )

        new_round_page = self.client.get(f"/classrooms/{classroom_id}/rounds/new")
        csrf = self._extract_csrf(new_round_page.get_data(as_text=True))
        response = self.client.post(
            f"/classrooms/{classroom_id}/rounds/new",
            data={"csrf_token": csrf, "win_weight": "0.7", "homework_weight": "0.3"},
            follow_redirects=True,
        )

        self.assertEqual(response.status_code, 200)
        body = response.get_data(as_text=True)
        self.assertIn("results", body)
        self.assertNotIn("We hit an unexpected error", body)



    def test_login_redirects_to_setup_when_no_teachers(self) -> None:
        response = self.client.get("/login", follow_redirects=False)
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.headers["Location"], "/setup")

    def test_setup_redirects_to_login_when_teacher_already_exists(self) -> None:
        self._bootstrap_teacher_and_login()

        response = self.client.get("/setup", follow_redirects=False)
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.headers["Location"], "/login")

    def test_signup_allows_second_teacher_creation(self) -> None:
        self._bootstrap_teacher_and_login()

        signup_page = self.client.get("/signup")
        self.assertEqual(signup_page.status_code, 200)
        csrf = self._extract_csrf(signup_page.get_data(as_text=True))

        response = self.client.post(
            "/signup",
            data={
                "csrf_token": csrf,
                "username": "teacher2",
                "password": "pw2",
                "confirm_password": "pw2",
            },
            follow_redirects=False,
        )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.headers["Location"], "/login")

        with self.db.session_scope() as db_session:
            teacher_count = db_session.query(self.app_module.Teacher).count()

        self.assertEqual(teacher_count, 2)

    def test_signup_duplicate_username_shows_validation_error(self) -> None:
        self._bootstrap_teacher_and_login()

        signup_page = self.client.get("/signup")
        csrf = self._extract_csrf(signup_page.get_data(as_text=True))

        response = self.client.post(
            "/signup",
            data={
                "csrf_token": csrf,
                "username": "teacher1",
                "password": "pw2",
                "confirm_password": "pw2",
            },
            follow_redirects=True,
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn("Username is already taken.", response.get_data(as_text=True))

    def _create_round_and_get_id(self, classroom_id: int) -> int:
        new_round_page = self.client.get(f"/classrooms/{classroom_id}/rounds/new")
        csrf = self._extract_csrf(new_round_page.get_data(as_text=True))
        response = self.client.post(
            f"/classrooms/{classroom_id}/rounds/new",
            data={
                "csrf_token": csrf,
                "win_weight": "0.7",
                "homework_weight": "0.3",
                "homework_total_questions": "10",
                "missing_homework_policy": "zero",
                "homework_metric_mode": "pct_correct",
            },
            follow_redirects=False,
        )
        self.assertEqual(response.status_code, 302)
        location = response.headers["Location"]
        match = re.search(r"/rounds/(\d+)", location)
        self.assertIsNotNone(match)
        return int(match.group(1))

    def _first_match_id(self, round_id: int) -> int:
        with self.db.session_scope() as db_session:
            match = (
                db_session.query(self.app_module.Match)
                .filter(self.app_module.Match.round_id == round_id)
                .order_by(self.app_module.Match.id)
                .first()
            )
        self.assertIsNotNone(match)
        return int(match.id)

    def test_export_round_includes_homework_notes_results_and_byes(self) -> None:
        self._bootstrap_teacher_and_login()
        classroom_id = self._create_classroom("Round Export")
        self._add_students(classroom_id, ["A", "B", "C"])
        round_id = self._create_round_and_get_id(classroom_id)
        match_id = self._first_match_id(round_id)

        round_page = self.client.get(f"/classrooms/{classroom_id}/rounds/{round_id}")
        csrf = self._extract_csrf(round_page.get_data(as_text=True))
        self.client.post(
            f"/classrooms/{classroom_id}/rounds/{round_id}",
            data={
                "csrf_token": csrf,
                "action": "save",
                "homework_total_questions": "10",
                "missing_homework_policy": "zero",
                "homework_metric_mode": "pct_correct",
                f"result_{match_id}": "white",
                f"notes_{match_id}": "good game",
                f"white_entered_correct_{match_id}": "8",
                f"black_entered_correct_{match_id}": "7",
            },
            follow_redirects=False,
        )

        export_response = self.client.get(f"/classrooms/{classroom_id}/rounds/{round_id}/export")
        self.assertEqual(export_response.status_code, 200)
        rows = list(csv.DictReader(io.StringIO(export_response.get_data(as_text=True))))
        self.assertTrue(rows)
        self.assertIn("White Homework Submitted", rows[0].keys())
        self.assertIn("White Notation Submitted", rows[0].keys())

    def test_autosave_persists_round_edits(self) -> None:
        self._bootstrap_teacher_and_login()
        classroom_id = self._create_classroom("Autosave")
        self._add_students(classroom_id, ["A", "B"])
        round_id = self._create_round_and_get_id(classroom_id)
        match_id = self._first_match_id(round_id)

        round_page = self.client.get(f"/classrooms/{classroom_id}/rounds/{round_id}")
        csrf = self._extract_csrf(round_page.get_data(as_text=True))

        autosave_response = self.client.post(
            f"/classrooms/{classroom_id}/rounds/{round_id}/autosave",
            data={
                "csrf_token": csrf,
                "match_id": str(match_id),
                "field": "notes",
                "value": "autosaved note",
            },
        )
        self.assertEqual(autosave_response.status_code, 200)
        self.assertIn('"saved":true', autosave_response.get_data(as_text=True).replace(" ", ""))

        refreshed = self.client.get(f"/classrooms/{classroom_id}/rounds/{round_id}")
        self.assertIn("autosaved note", refreshed.get_data(as_text=True))

    def test_homework_policy_and_completion_guardrail(self) -> None:
        self._bootstrap_teacher_and_login()
        classroom_id = self._create_classroom("Policy")
        self._add_students(classroom_id, ["A", "B"])
        round_id = self._create_round_and_get_id(classroom_id)
        match_id = self._first_match_id(round_id)

        round_page = self.client.get(f"/classrooms/{classroom_id}/rounds/{round_id}")
        csrf = self._extract_csrf(round_page.get_data(as_text=True))
        blocked_complete = self.client.post(
            f"/classrooms/{classroom_id}/rounds/{round_id}",
            data={
                "csrf_token": csrf,
                "action": "complete_round",
                "homework_total_questions": "10",
                "missing_homework_policy": "zero",
                "homework_metric_mode": "pct_correct",
                f"white_entered_correct_{match_id}": "",
                f"black_entered_correct_{match_id}": "",
                f"result_{match_id}": "",
            },
            follow_redirects=True,
        )
        self.assertIn("Round has unresolved data", blocked_complete.get_data(as_text=True))

        complete_with_override = self.client.post(
            f"/classrooms/{classroom_id}/rounds/{round_id}",
            data={
                "csrf_token": csrf,
                "action": "complete_round",
                "override_reason": "Teacher approved incomplete entries",
                "homework_total_questions": "10",
                "missing_homework_policy": "zero",
                "homework_metric_mode": "pct_correct",
                f"result_{match_id}": "white",
            },
            follow_redirects=True,
        )
        self.assertEqual(complete_with_override.status_code, 200)
        self.assertIn("Round status: <span class=\"font-semibold\">completed</span>", complete_with_override.get_data(as_text=True))

    def test_complete_round_handles_missing_policy_fields_without_500(self) -> None:
        self._bootstrap_teacher_and_login()
        classroom_id = self._create_classroom("Null-safe Complete")
        self._add_students(classroom_id, ["A", "B"])
        round_id = self._create_round_and_get_id(classroom_id)
        match_id = self._first_match_id(round_id)

        round_page = self.client.get(f"/classrooms/{classroom_id}/rounds/{round_id}")
        csrf = self._extract_csrf(round_page.get_data(as_text=True))
        response = self.client.post(
            f"/classrooms/{classroom_id}/rounds/{round_id}",
            data={
                "csrf_token": csrf,
                "action": "complete_round",
                "homework_total_questions": "10",
                f"result_{match_id}": "white",
            },
            follow_redirects=True,
        )

        body = response.get_data(as_text=True)
        self.assertEqual(response.status_code, 200)
        self.assertNotIn("We hit an unexpected error", body)

    def test_round_results_handles_sqlalchemy_error_without_500(self) -> None:
        self._bootstrap_teacher_and_login()
        classroom_id = self._create_classroom("Recoverable DB Error")
        self._add_students(classroom_id, ["A", "B"])
        round_id = self._create_round_and_get_id(classroom_id)
        match_id = self._first_match_id(round_id)

        round_page = self.client.get(f"/classrooms/{classroom_id}/rounds/{round_id}")
        csrf = self._extract_csrf(round_page.get_data(as_text=True))

        with patch("app.recalculate_totals", side_effect=self.app_module.SQLAlchemyError("forced")):
            response = self.client.post(
                f"/classrooms/{classroom_id}/rounds/{round_id}",
                data={
                    "csrf_token": csrf,
                    "action": "save",
                    "homework_total_questions": "10",
                    "missing_homework_policy": "zero",
                    "homework_metric_mode": "pct_correct",
                    f"result_{match_id}": "white",
                },
                follow_redirects=True,
            )

        body = response.get_data(as_text=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn("We couldn't save round changes due to a temporary data issue. Please retry.", body)
        self.assertNotIn("We hit an unexpected error", body)

    def test_complete_round_handles_flush_sqlalchemy_error_without_500(self) -> None:
        self._bootstrap_teacher_and_login()
        classroom_id = self._create_classroom("Complete Flush Error")
        self._add_students(classroom_id, ["A", "B"])
        round_id = self._create_round_and_get_id(classroom_id)
        match_id = self._first_match_id(round_id)

        round_page = self.client.get(f"/classrooms/{classroom_id}/rounds/{round_id}")
        csrf = self._extract_csrf(round_page.get_data(as_text=True))

        with patch(
            "sqlalchemy.orm.session.Session.flush",
            side_effect=self.app_module.SQLAlchemyError("forced"),
        ):
            response = self.client.post(
                f"/classrooms/{classroom_id}/rounds/{round_id}",
                data={
                    "csrf_token": csrf,
                    "action": "complete_round",
                    "override_reason": "manual override",
                    "homework_total_questions": "10",
                    "missing_homework_policy": "zero",
                    "homework_metric_mode": "pct_correct",
                    f"result_{match_id}": "white",
                },
                follow_redirects=True,
            )

        body = response.get_data(as_text=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn("We couldn't save round changes due to a temporary data issue. Please retry.", body)
        self.assertNotIn("We hit an unexpected error", body)

    def test_health_db_includes_schema_compatibility_fields(self) -> None:
        self._bootstrap_teacher_and_login()
        response = self.client.get("/health/db")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertIn("schema_compatibility", payload)
        self.assertIn("schema_issues", payload)
        self.assertIsInstance(payload["schema_issues"], list)
        self.assertIn(payload["schema_compatibility"], {"ok", "issues"})

    def test_student_export_import_roundtrip_contract(self) -> None:
        self._bootstrap_teacher_and_login()
        classroom_id = self._create_classroom("Roundtrip")
        self._add_students(classroom_id, ["A", "B", "C", "D"])

        export_response = self.client.get(f"/classrooms/{classroom_id}/export/students")
        self.assertEqual(export_response.status_code, 200)
        exported_csv = export_response.get_data()

        import_page = self.client.get(f"/classrooms/{classroom_id}/import")
        csrf = self._extract_csrf(import_page.get_data(as_text=True))
        import_response = self.client.post(
            f"/classrooms/{classroom_id}/import",
            data={
                "csrf_token": csrf,
                "replace_existing": "on",
                "file": (io.BytesIO(exported_csv), "students_roundtrip.csv"),
            },
            content_type="multipart/form-data",
            follow_redirects=False,
        )
        self.assertEqual(import_response.status_code, 302)


class SchemaVerificationTests(unittest.TestCase):
    def test_verify_schema_compatibility_fail_fast_raises(self) -> None:
        db = importlib.import_module("db")
        with patch("db.schema_compatibility_issues", return_value=["missing column: students.active"]):
            with self.assertRaises(RuntimeError):
                db.verify_schema_compatibility(fail_fast=True)


if __name__ == "__main__":
    unittest.main()
