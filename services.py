from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from pairing_logic import normalize_weights, select_pairings
from models import AssignmentEntry, HomeworkEntry, Match, Student, StudentAssignmentScore

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
    homework_weight: float = 0.0,
    homework_metric_mode: str = "pct_correct",
    assignment_type_data: Optional[List[Dict]] = None,
    missing_score_pct: Optional[float] = None,
) -> Tuple[pd.DataFrame, List[int]]:
    """Build a rating DataFrame for pairing.

    When *assignment_type_data* is provided it is used instead of the legacy
    homework_weight / homework_metric_mode path.  Each element is a dict with:
      - ``weight``        float (pre-normalised fraction of total rating)
      - ``metric_mode``   str (``'pct_correct'`` or ``'total_correct'``)
      - ``student_scores`` dict mapping student_id -> (correct: int, incorrect: int)

    *missing_score_pct*: None = treat students with no score as 0 (excluded effect),
    0-100 = substitute that percentage for students absent from *student_scores*.
    """
    rows: List[RatingRow] = []
    student_cache: List[Student] = list(students)
    metric_mode = canonicalize_homework_metric_mode(homework_metric_mode)

    def safe_int(value: Optional[int]) -> int:
        if value is None:
            return 0
        return int(value)

    if assignment_type_data:
        # ── New multi-assignment-type path ──────────────────────────────────────
        # Pre-compute per-type score lists (one entry per student, same order as student_cache).
        at_score_lists: List[List[float]] = []
        for at in assignment_type_data:
            at_metric = canonicalize_homework_metric_mode(at.get("metric_mode", "pct_correct"))
            scores_map: Dict[int, Tuple[int, int]] = at.get("student_scores", {})
            at_missing_pct = at.get("missing_score_pct", missing_score_pct)  # per-type, fallback to global
            missing_indices: List[int] = []
            raw: List[float] = []
            for idx, student in enumerate(student_cache):
                if student.id not in scores_map:
                    missing_indices.append(idx)
                correct, incorrect = scores_map.get(student.id, (0, 0))
                total = correct + incorrect
                if at_metric == "total_correct":
                    raw.append(float(correct))
                else:
                    raw.append(correct / total if total else 0.0)
            # Normalise total_correct scores by the class maximum.
            if at_metric == "total_correct":
                max_val = max(raw) if raw else 0.0
                raw = [s / max_val if max_val > 0 else 0.0 for s in raw]
            # Apply missing_score_pct for students with no historical score.
            if at_missing_pct is not None and missing_indices:
                synthetic = max(0.0, min(100.0, at_missing_pct)) / 100.0
                for idx in missing_indices:
                    raw[idx] = synthetic
            at_score_lists.append(raw)

        for idx, student in enumerate(student_cache):
            total_wins = safe_int(student.total_wins)
            total_losses = safe_int(student.total_losses)
            total_ties = safe_int(student.total_ties)
            total_games = total_wins + total_losses + total_ties
            win_rate = total_wins / total_games if total_games else 0
            blended = sum(at["weight"] * at_score_lists[i][idx] for i, at in enumerate(assignment_type_data))
            rating = round(win_weight * win_rate + blended, 3)
            color_diff = safe_int(student.times_white) - safe_int(student.times_black)
            rows.append(RatingRow(student_id=student.id, name=student.name, rating=rating, color_diff=color_diff))
    else:
        # ── Legacy single-homework path ─────────────────────────────────────────
        homework_components: List[float] = []
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
    homework_weight: float = 0.0,
    homework_metric_mode: str = "pct_correct",
    assignment_type_data: Optional[List[Dict]] = None,
    missing_score_pct: Optional[float] = None,
) -> Tuple[List[Tuple[int, int]], List[int], pd.DataFrame, List[int]]:
    if assignment_type_data is not None:
        all_weights = [win_weight] + [at["weight"] for at in assignment_type_data]
        normalized = normalize_weights(all_weights)
        normalized_win = normalized[0]
        normalized_ats = [
            {**at, "weight": normalized[i + 1]} for i, at in enumerate(assignment_type_data)
        ]
        df, id_order = build_rating_dataframe(
            students, normalized_win, assignment_type_data=normalized_ats,
            missing_score_pct=missing_score_pct,
        )
    else:
        normalized_win, normalized_homework = normalize_weights(win_weight, homework_weight)
        df, id_order = build_rating_dataframe(
            students, normalized_win, normalized_homework, homework_metric_mode,
            missing_score_pct=missing_score_pct,
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


def recalculate_totals(students: Iterable[Student], matches: Iterable[Match], db=None) -> None:
    student_map = {student.id: student for student in students}
    for student in student_map.values():
        student.total_wins = 0
        student.total_losses = 0
        student.total_ties = 0
        student.times_white = 0
        student.times_black = 0
        student.homework_correct = 0
        student.homework_incorrect = 0

    # Accumulator for the new assignment-type scores.
    # {(student_id, assignment_type_id): [correct, incorrect]}
    assignment_accum: Dict[Tuple[int, int], List[int]] = {}

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

        # Legacy HomeworkEntry — keeps Student.homework_correct/incorrect populated for
        # backward-compat with CSV export and the legacy rating path.
        if homework:
            if white_id and white_id in student_map and not homework.white_exempt:
                student_map[white_id].homework_correct += int(homework.white_correct or 0)
                student_map[white_id].homework_incorrect += int(homework.white_incorrect or 0)
            if black_id and black_id in student_map and not homework.black_exempt:
                student_map[black_id].homework_correct += int(homework.black_correct or 0)
                student_map[black_id].homework_incorrect += int(homework.black_incorrect or 0)

        # New AssignmentEntry records — accumulate per (student, assignment_type).
        for entry in (getattr(match, "assignment_entries", None) or []):
            at_id = entry.assignment_type_id
            if white_id and white_id in student_map and not entry.white_exempt:
                key = (white_id, at_id)
                acc = assignment_accum.setdefault(key, [0, 0])
                acc[0] += int(entry.white_correct or 0)
                acc[1] += int(entry.white_incorrect or 0)
            if black_id and black_id in student_map and not entry.black_exempt:
                key = (black_id, at_id)
                acc = assignment_accum.setdefault(key, [0, 0])
                acc[0] += int(entry.black_correct or 0)
                acc[1] += int(entry.black_incorrect or 0)

    # Persist StudentAssignmentScore when a DB session is provided.
    if db is not None:
        student_ids = list(student_map.keys())
        if student_ids:
            db.query(StudentAssignmentScore).filter(
                StudentAssignmentScore.student_id.in_(student_ids)
            ).delete(synchronize_session="fetch")
        for (student_id, at_id), (correct, incorrect) in assignment_accum.items():
            db.add(
                StudentAssignmentScore(
                    student_id=student_id,
                    assignment_type_id=at_id,
                    correct=correct,
                    incorrect=incorrect,
                )
            )


def apply_homework_policy(
    entered_correct: Optional[str],
    total_questions: int,
    missing_score_pct: Optional[float],
) -> tuple[int, int, bool, float]:
    """Compute correct/incorrect/submitted/pct_wrong from round policy.

    missing_score_pct:
      None        – exclude blank entries (no effect on the student's aggregate)
      0 – 100     – treat blank entries as that percentage correct
    """
    entered_value = (entered_correct or "").strip()
    if not entered_value:
        if missing_score_pct is None:
            return 0, 0, False, 0.0
        pct = max(0.0, min(100.0, missing_score_pct)) / 100.0
        correct = round(pct * max(total_questions, 0))
        wrong = max(total_questions - correct, 0)
        pct_wrong = (wrong / total_questions) if total_questions > 0 else 0.0
        return correct, wrong, False, pct_wrong

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


def compute_classroom_analytics(classroom_id: int, db: Any) -> Dict[str, Any]:
    """Build the full analytics data dict for the analytics page.

    Returns a dict with keys:
      - ``rounds``           list of round numbers (ints) for all completed rounds
      - ``round_ids``        parallel list of round DB ids
      - ``students``         list of per-student dicts (see below)
      - ``assignment_types`` list of {id, name, analytics_weight}
      - ``analytics_win_weight`` int

    Per-student dict keys:
      ``id``, ``name``,
      ``chess``       – {round_number: {wins, losses, ties, bye}},
      ``attendance``  – {round_number: status string (or None)},
      ``assignments`` – {at_id: {round_number: {correct, incorrect, pct}}},
      ``strength``    – {round_number: float}  (NaN-safe, None if not computable)
    """
    from models import Attendance, AssignmentType, Classroom, Round, Match, AssignmentEntry
    from sqlalchemy.orm import selectinload

    classroom: Classroom = db.query(Classroom).filter(Classroom.id == classroom_id).first()
    if classroom is None:
        return {"rounds": [], "round_ids": [], "students": [], "assignment_types": [], "analytics_win_weight": 50}

    analytics_win_weight: int = getattr(classroom, "analytics_win_weight", 50) or 50

    # Fetch assignment types for this classroom
    assignment_types_raw = classroom.assignment_types  # already ordered by id
    at_list = [
        {
            "id": at.id,
            "name": at.name,
            "analytics_weight": getattr(at, "analytics_weight", 50) or 50,
        }
        for at in assignment_types_raw
    ]

    # Fetch all completed rounds ordered by round_number / id
    rounds_q = (
        db.query(Round)
        .filter(Round.classroom_id == classroom_id, Round.status == "completed")
        .order_by(Round.round_number.nullslast(), Round.id)
        .all()
    )
    round_numbers = [
        r.round_number if r.round_number is not None else r.id for r in rounds_q
    ]
    round_ids = [r.id for r in rounds_q]

    if not rounds_q:
        students_out = []
        for student in classroom.students:
            if student.active:
                students_out.append(_empty_student_entry(student, at_list))
        return {
            "rounds": round_numbers,
            "round_ids": round_ids,
            "students": students_out,
            "assignment_types": at_list,
            "analytics_win_weight": analytics_win_weight,
        }

    # Fetch all matches for these rounds (with assignment entries eager-loaded)
    all_matches = (
        db.query(Match)
        .filter(Match.round_id.in_(round_ids))
        .options(selectinload(Match.assignment_entries))
        .all()
    )

    # Fetch all attendance for these rounds
    all_attendance = (
        db.query(Attendance)
        .filter(Attendance.round_id.in_(round_ids))
        .all()
    )

    # Index by round_id for quick lookup
    round_id_to_number = {r.id: (r.round_number if r.round_number is not None else r.id) for r in rounds_q}

    # Build per-student data structures
    active_students = [s for s in classroom.students if s.active]

    # {student_id: {round_number: {"wins": int, "losses": int, "ties": int, "bye": bool}}}
    chess_data: Dict[int, Dict[int, Dict[str, Any]]] = {s.id: {} for s in active_students}
    # {student_id: {round_number: status_str | None}}
    attendance_data: Dict[int, Dict[int, Optional[str]]] = {s.id: {} for s in active_students}
    # {student_id: {at_id: {round_number: {correct, incorrect, pct}}}}
    assign_data: Dict[int, Dict[int, Dict[int, Dict[str, Any]]]] = {
        s.id: {at["id"]: {} for at in at_list} for s in active_students
    }

    student_ids = {s.id for s in active_students}

    for match in all_matches:
        rnum = round_id_to_number.get(match.round_id)
        if rnum is None:
            continue
        result = (match.result or "").lower()
        w_id = match.white_student_id
        b_id = match.black_student_id

        def _ensure_chess(sid: int) -> None:
            if sid in chess_data and rnum not in chess_data[sid]:
                chess_data[sid][rnum] = {"wins": 0, "losses": 0, "ties": 0, "bye": False}

        if w_id and w_id in student_ids:
            _ensure_chess(w_id)
            if result == "white" and b_id:
                # Count win even if the opponent has since been deactivated
                chess_data[w_id][rnum]["wins"] += 1
            elif result == "black" and b_id:
                # Count loss even if the opponent has since been deactivated
                chess_data[w_id][rnum]["losses"] += 1
            elif result == "tie" and b_id:
                chess_data[w_id][rnum]["ties"] += 1
            elif result == "bye":
                chess_data[w_id][rnum]["bye"] = True

        if b_id and b_id in student_ids:
            _ensure_chess(b_id)
            if result == "black":
                chess_data[b_id][rnum]["wins"] += 1
            elif result == "white" and w_id:
                # Count loss even if the opponent has since been deactivated
                chess_data[b_id][rnum]["losses"] += 1
            elif result == "tie":
                chess_data[b_id][rnum]["ties"] += 1

        # Assignment entries
        for entry in (match.assignment_entries or []):
            at_id = entry.assignment_type_id
            if w_id and w_id in student_ids and at_id in assign_data.get(w_id, {}):
                _w_has_score = (int(entry.white_correct or 0) + int(entry.white_incorrect or 0)) > 0
                if not entry.white_exempt and (entry.white_submitted or _w_has_score):
                    _acc = assign_data[w_id][at_id].setdefault(rnum, {"correct": 0, "incorrect": 0})
                    _acc["correct"] += int(entry.white_correct or 0)
                    _acc["incorrect"] += int(entry.white_incorrect or 0)
            if b_id and b_id in student_ids and at_id in assign_data.get(b_id, {}):
                _b_has_score = (int(entry.black_correct or 0) + int(entry.black_incorrect or 0)) > 0
                if not entry.black_exempt and (entry.black_submitted or _b_has_score):
                    _acc = assign_data[b_id][at_id].setdefault(rnum, {"correct": 0, "incorrect": 0})
                    _acc["correct"] += int(entry.black_correct or 0)
                    _acc["incorrect"] += int(entry.black_incorrect or 0)

    # Attendance
    for rec in all_attendance:
        rnum = round_id_to_number.get(rec.round_id)
        if rnum is not None and rec.student_id in attendance_data:
            attendance_data[rec.student_id][rnum] = rec.status

    # Compute pct for assignment entries and strength scores
    # Normalise analytics weights
    raw_weights = [analytics_win_weight] + [at["analytics_weight"] for at in at_list]
    weight_sum = sum(raw_weights) or 1
    norm_win = analytics_win_weight / weight_sum
    norm_at_weights = [at["analytics_weight"] / weight_sum for at in at_list]

    students_out = []
    for student in active_students:
        sid = student.id
        # Finalise assignment pct
        for at in at_list:
            at_id = at["id"]
            for rnum, acc in assign_data[sid][at_id].items():
                total = acc["correct"] + acc["incorrect"]
                acc["pct"] = round(acc["correct"] / total * 100, 1) if total else None

        # Compute strength score per round
        strength: Dict[int, Optional[float]] = {}
        for rnum in round_numbers:
            c_data = chess_data[sid].get(rnum, {"wins": 0, "losses": 0, "ties": 0})
            games = c_data["wins"] + c_data["losses"] + c_data["ties"]
            win_rate = c_data["wins"] / games if games else 0.0
            score = norm_win * win_rate
            for i, at in enumerate(at_list):
                at_id = at["id"]
                rnd_assign = assign_data[sid][at_id].get(rnum)
                if rnd_assign and rnd_assign.get("pct") is not None:
                    score += norm_at_weights[i] * (rnd_assign["pct"] / 100.0)
            strength[rnum] = round(score * 100, 1)

        students_out.append({
            "id": sid,
            "name": student.name,
            "chess": chess_data[sid],
            "attendance": attendance_data[sid],
            "assignments": assign_data[sid],
            "strength": strength,
        })

    return {
        "rounds": round_numbers,
        "round_ids": round_ids,
        "students": students_out,
        "assignment_types": at_list,
        "analytics_win_weight": analytics_win_weight,
    }


def _empty_student_entry(student: Student, at_list: List[Dict]) -> Dict[str, Any]:
    return {
        "id": student.id,
        "name": student.name,
        "chess": {},
        "attendance": {},
        "assignments": {at["id"]: {} for at in at_list},
        "strength": {},
    }
