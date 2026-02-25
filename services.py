from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, List, Optional, Tuple

import pandas as pd

from pairing_logic import normalize_weights, select_pairings
from models import HomeworkEntry, Match, Round, Student


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
    homework_metric_mode: str = "pct_wrong",
) -> Tuple[pd.DataFrame, List[int]]:
    students = list(students)
    rows: List[RatingRow] = []

    def safe_int(value: Optional[int]) -> int:
        if value is None:
            return 0
        return int(value)

    max_homework_correct = max(
        (safe_int(student.homework_correct) for student in students),
        default=0,
    )

    for student in students:
        total_wins = safe_int(student.total_wins)
        total_losses = safe_int(student.total_losses)
        total_ties = safe_int(student.total_ties)
        total_games = total_wins + total_losses + total_ties
        win_rate = total_wins / total_games if total_games else 0
        homework_correct = safe_int(student.homework_correct)
        homework_incorrect = safe_int(student.homework_incorrect)
        total_homework = homework_correct + homework_incorrect
        homework_score = homework_correct / total_homework if total_homework else 0
        if homework_metric_mode == "raw_counts":
            homework_score = (
                homework_correct / max_homework_correct if max_homework_correct else 0
            )
        elif homework_metric_mode == "pct_correct":
            homework_score = homework_correct / total_homework if total_homework else 0
        elif homework_metric_mode == "pct_wrong":
            homework_score = 1 - (homework_incorrect / total_homework) if total_homework else 0
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
    homework_metric_mode: str = "pct_wrong",
) -> Tuple[List[Tuple[int, int]], List[int], pd.DataFrame, List[int]]:
    normalized_win, normalized_homework = normalize_weights(win_weight, homework_weight)
    df, id_order = build_rating_dataframe(
        students,
        normalized_win,
        normalized_homework,
        homework_metric_mode=homework_metric_mode,
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
        round_record: Optional[Round] = match.round

        if white_id and white_id in student_map:
            student_map[white_id].times_white += 1
        if black_id and black_id in student_map:
            student_map[black_id].times_black += 1

        if result == "white" and white_id and black_id:
            student_map[white_id].total_wins += 1
            student_map[black_id].total_losses += 1
        elif result == "black" and white_id and black_id:
            student_map[black_id].total_wins += 1
            student_map[white_id].total_losses += 1
        elif result == "tie" and white_id and black_id:
            student_map[white_id].total_ties += 1
            student_map[black_id].total_ties += 1

        if homework:
            total_questions = round_record.homework_total_questions if round_record else 0
            missing_policy = round_record.missing_homework_policy if round_record else "zero"
            missing_penalty = round_record.missing_homework_penalty if round_record else 1

            def apply_homework(
                student_id: Optional[int],
                correct: int,
                incorrect: int,
                submitted: bool,
            ) -> None:
                if not student_id or student_id not in student_map:
                    return
                if submitted:
                    student_map[student_id].homework_correct += int(correct)
                    student_map[student_id].homework_incorrect += int(incorrect)
                    return
                if total_questions <= 0:
                    return
                if missing_policy == "exclude":
                    return
                if missing_policy == "penalty":
                    student_map[student_id].homework_incorrect += max(1, int(missing_penalty))
                else:
                    student_map[student_id].homework_incorrect += int(total_questions)

            apply_homework(
                white_id,
                homework.white_correct,
                homework.white_incorrect,
                homework.white_submitted,
            )
            apply_homework(
                black_id,
                homework.black_correct,
                homework.black_incorrect,
                homework.black_submitted,
            )
