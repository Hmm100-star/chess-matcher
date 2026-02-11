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


class SchemaVerificationTests(unittest.TestCase):
    def test_verify_schema_compatibility_fail_fast_raises(self) -> None:
        db = importlib.import_module("db")
        with patch("db.schema_compatibility_issues", return_value=["missing column: students.active"]):
            with self.assertRaises(RuntimeError):
                db.verify_schema_compatibility(fail_fast=True)


if __name__ == "__main__":
    unittest.main()
