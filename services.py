from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, List, Optional, Tuple

import pandas as pd

from pairing_logic import normalize_weights, select_pairings
from models import HomeworkEntry, Match, Student

VALID_HOMEWORK_METRIC_MODES = {"pct_correct", "total_correct", "pct_wrong", "raw_counts"}


def canonicalize_homework_metric_mode(metric_mode: Optional[str]) -> str:
    normalized = (metric_mode or "").strip()
    if normalized in {"pct_correct", "pct_wrong"}:
        return "pct_correct"
    if normalized in {"total_correct", "raw_counts"}:
        return "total_correct"
    return "pct_correct"


@dataclass
class RatingRow:
    student_id: int
    name: str
    rating: float
    color_diff: int


def build_rating_dataframe(
    students: Iterable[Student],
    win_weight: float,
    homework_weight: float,
    homework_metric_mode: str = "pct_correct",
) -> Tuple[pd.DataFrame, List[int]]:
    rows: List[RatingRow] = []
    homework_components: List[float] = []
    student_cache: List[Student] = list(students)
    metric_mode = canonicalize_homework_metric_mode(homework_metric_mode)

    def safe_int(value: Optional[int]) -> int:
        if value is None:
            return 0
        return int(value)

    for student in student_cache:
        homework_correct = safe_int(student.homework_correct)
        homework_incorrect = safe_int(student.homework_incorrect)
        total_homework = homework_correct + homework_incorrect
        if metric_mode == "total_correct":
            homework_components.append(float(homework_correct))
        else:
            homework_components.append(homework_correct / total_homework if total_homework else 0)

    max_raw_homework = max(homework_components) if homework_components else 1.0
    if max_raw_homework <= 0:
        max_raw_homework = 1.0

    for index, student in enumerate(student_cache):
        total_wins = safe_int(student.total_wins)
        total_losses = safe_int(student.total_losses)
        total_ties = safe_int(student.total_ties)
        total_games = total_wins + total_losses + total_ties
        win_rate = total_wins / total_games if total_games else 0
        homework_score = homework_components[index]
        if metric_mode == "total_correct":
            homework_score = homework_score / max_raw_homework
        rating = round((win_weight * win_rate) + (homework_weight * homework_score), 3)
        color_diff = safe_int(student.times_white) - safe_int(student.times_black)
        rows.append(
            RatingRow(
                student_id=student.id,
                name=student.name,
                rating=rating,
                color_diff=color_diff,
            )
        )

    df = pd.DataFrame(
        [
            {
                "student_id": row.student_id,
                "Student Name": row.name,
                "rating": row.rating,
                "color_diff": row.color_diff,
            }
            for row in rows
        ]
    )
    df = df.sort_values(by=["rating", "color_diff"], ascending=[False, True]).reset_index(
        drop=True
    )
    id_order = df["student_id"].tolist()
    return df, id_order


def generate_matches_for_students(
    students: Iterable[Student],
    win_weight: float,
    homework_weight: float,
    homework_metric_mode: str = "pct_correct",
) -> Tuple[List[Tuple[int, int]], List[int], pd.DataFrame, List[int]]:
    normalized_win, normalized_homework = normalize_weights(win_weight, homework_weight)
    df, id_order = build_rating_dataframe(
        students, normalized_win, normalized_homework, homework_metric_mode
    )
    matches, unpaired = select_pairings(df, rng=_random())
    return matches, unpaired, df, id_order


def _random():
    import random

    return random.Random()


def create_match_records(
    students: List[Student],
    matches: List[Tuple[int, int]],
    unpaired: List[int],
    df: pd.DataFrame,
    id_order: List[int],
) -> List[Match]:
    index_to_student_id = {idx: id_order[idx] for idx in range(len(id_order))}
    records: List[Match] = []

    for white_idx, black_idx in matches:
        white_id = index_to_student_id[white_idx]
        black_id = index_to_student_id[black_idx]
        records.append(
            Match(
                white_student_id=white_id,
                black_student_id=black_id,
                white_strength=f"{df.at[white_idx, 'rating']:.3f}",
                black_strength=f"{df.at[black_idx, 'rating']:.3f}",
                result=None,
                notes="",
                updated_at=datetime.utcnow(),
            )
        )

    for index in unpaired:
        student_id = index_to_student_id[index]
        records.append(
            Match(
                white_student_id=student_id,
                black_student_id=None,
                white_strength=f"{df.at[index, 'rating']:.3f}",
                black_strength=None,
                result="bye",
                notes="",
                updated_at=datetime.utcnow(),
            )
        )

    return records


def recalculate_totals(students: Iterable[Student], matches: Iterable[Match]) -> None:
    student_map = {student.id: student for student in students}
    for student in student_map.values():
        student.total_wins = 0
        student.total_losses = 0
        student.total_ties = 0
        student.times_white = 0
        student.times_black = 0
        student.homework_correct = 0
        student.homework_incorrect = 0

    for match in matches:
        white_id = match.white_student_id
        black_id = match.black_student_id
        result = (match.result or "").lower()
        homework: Optional[HomeworkEntry] = match.homework_entry

        if white_id and white_id in student_map:
            student_map[white_id].times_white += 1
        if black_id and black_id in student_map:
            student_map[black_id].times_black += 1

        if result == "white" and white_id and black_id and white_id in student_map and black_id in student_map:
            student_map[white_id].total_wins += 1
            student_map[black_id].total_losses += 1
        elif result == "black" and white_id and black_id and white_id in student_map and black_id in student_map:
            student_map[black_id].total_wins += 1
            student_map[white_id].total_losses += 1
        elif result == "tie" and white_id and black_id and white_id in student_map and black_id in student_map:
            student_map[white_id].total_ties += 1
            student_map[black_id].total_ties += 1

        if homework:
            if white_id and white_id in student_map and not homework.white_exempt:
                student_map[white_id].homework_correct += int(homework.white_correct or 0)
                student_map[white_id].homework_incorrect += int(homework.white_incorrect or 0)
            if black_id and black_id in student_map and not homework.black_exempt:
                student_map[black_id].homework_correct += int(homework.black_correct or 0)
                student_map[black_id].homework_incorrect += int(homework.black_incorrect or 0)


def apply_homework_policy(
    entered_correct: Optional[str],
    total_questions: int,
    missing_policy: str,
) -> tuple[int, int, bool, float]:
    """Compute correct/incorrect/submitted/pct_wrong from round policy."""
    entered_value = (entered_correct or "").strip()
    if not entered_value:
        if missing_policy == "exclude":
            return 0, 0, False, 0.0
        if missing_policy == "penalty":
            wrong = max(total_questions, 0)
            return 0, wrong, False, 1.0 if wrong else 0.0
        wrong = max(total_questions, 0)
        return 0, wrong, False, 1.0 if wrong else 0.0

    parsed_correct = int(float(entered_value))
    if parsed_correct < 0:
        raise ValueError("Homework correct count cannot be negative.")
    if total_questions > 0 and parsed_correct > total_questions:
        raise ValueError("Homework correct count cannot exceed total questions.")
    if total_questions <= 0:
        wrong = 0
    else:
        wrong = max(total_questions - parsed_correct, 0)
    pct_wrong = (wrong / total_questions) if total_questions > 0 else 0.0
    return parsed_correct, wrong, True, pct_wrong
