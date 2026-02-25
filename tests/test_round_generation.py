from __future__ import annotations

import io
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

    def _create_round(self, classroom_id: int) -> int:
        new_round_page = self.client.get(f"/classrooms/{classroom_id}/rounds/new")
        csrf = self._extract_csrf(new_round_page.get_data(as_text=True))
        response = self.client.post(
            f"/classrooms/{classroom_id}/rounds/new",
            data={"csrf_token": csrf, "win_weight": "0.7", "homework_weight": "0.3"},
            follow_redirects=False,
        )
        self.assertEqual(response.status_code, 302)
        location = response.headers.get("Location", "")
        match = re.search(r"/rounds/(\d+)", location)
        self.assertIsNotNone(match)
        return int(match.group(1))

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
        self.assertIn("white_correct_", body)
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

    def test_round_export_includes_homework_and_notation(self) -> None:
        self._bootstrap_teacher_and_login()
        classroom_id = self._create_classroom("Class Export")
        self._add_students(classroom_id, ["A", "B", "C", "D"])
        round_id = self._create_round(classroom_id)

        round_page = self.client.get(f"/classrooms/{classroom_id}/rounds/{round_id}")
        csrf = self._extract_csrf(round_page.get_data(as_text=True))
        body = round_page.get_data(as_text=True)
        match_ids = re.findall(r'data-match-id="(\d+)"', body)
        self.assertTrue(match_ids)
        first = match_ids[0]

        submit_data = {
            "csrf_token": csrf,
            "action": "save",
            f"result_{first}": "white",
            f"white_submitted_{first}": "1",
            f"black_submitted_{first}": "1",
            f"white_correct_{first}": "8",
            f"black_correct_{first}": "6",
            f"notation_white_{first}": "1",
            f"notation_black_{first}": "1",
            f"notes_{first}": "Test note",
        }
        self.client.post(
            f"/classrooms/{classroom_id}/rounds/{round_id}",
            data=submit_data,
            follow_redirects=True,
        )

        export_response = self.client.get(
            f"/classrooms/{classroom_id}/rounds/{round_id}/export",
            follow_redirects=False,
        )
        self.assertEqual(export_response.status_code, 200)
        csv_text = export_response.get_data(as_text=True)
        self.assertIn("White Notation Submitted", csv_text)
        self.assertIn("Test note", csv_text)

    def test_round_autosave_persists_notes(self) -> None:
        self._bootstrap_teacher_and_login()
        classroom_id = self._create_classroom("Class Autosave")
        self._add_students(classroom_id, ["A", "B", "C", "D"])
        round_id = self._create_round(classroom_id)

        round_page = self.client.get(f"/classrooms/{classroom_id}/rounds/{round_id}")
        csrf = self._extract_csrf(round_page.get_data(as_text=True))
        body = round_page.get_data(as_text=True)
        match_ids = re.findall(r'data-match-id="(\d+)"', body)
        self.assertTrue(match_ids)
        first = match_ids[0]

        autosave_response = self.client.post(
            f"/classrooms/{classroom_id}/rounds/{round_id}/autosave",
            data={
                "csrf_token": csrf,
                "match_id": first,
                f"result_{first}": "tie",
                f"white_submitted_{first}": "1",
                f"black_submitted_{first}": "1",
                f"white_correct_{first}": "7",
                f"black_correct_{first}": "5",
                f"notes_{first}": "Autosaved note",
                f"notation_white_{first}": "1",
                f"notation_black_{first}": "1",
            },
            follow_redirects=False,
        )
        self.assertEqual(autosave_response.status_code, 200)

        refreshed = self.client.get(f"/classrooms/{classroom_id}/rounds/{round_id}")
        self.assertIn("Autosaved note", refreshed.get_data(as_text=True))

    def test_students_export_import_roundtrip(self) -> None:
        self._bootstrap_teacher_and_login()
        classroom_a = self._create_classroom("Export Source")
        self._add_students(classroom_a, ["A", "B"])

        export_response = self.client.get(f"/classrooms/{classroom_a}/export/students")
        self.assertEqual(export_response.status_code, 200)
        exported = export_response.data

        classroom_b = self._create_classroom("Import Dest")
        import_page = self.client.get(f"/classrooms/{classroom_b}/import")
        csrf = self._extract_csrf(import_page.get_data(as_text=True))
        import_response = self.client.post(
            f"/classrooms/{classroom_b}/import",
            data={
                "csrf_token": csrf,
                "replace_existing": "on",
                "file": (io.BytesIO(exported), "students_roundtrip.csv"),
            },
            content_type="multipart/form-data",
            follow_redirects=False,
        )
        self.assertEqual(import_response.status_code, 302)
        classroom_page = self.client.get(f"/classrooms/{classroom_b}")
        page = classroom_page.get_data(as_text=True)
        self.assertIn("A", page)
        self.assertIn("B", page)



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


class SchemaVerificationTests(unittest.TestCase):
    def test_verify_schema_compatibility_fail_fast_raises(self) -> None:
        db = importlib.import_module("db")
        with patch("db.schema_compatibility_issues", return_value=["missing column: students.active"]):
            with self.assertRaises(RuntimeError):
                db.verify_schema_compatibility(fail_fast=True)


if __name__ == "__main__":
    unittest.main()
