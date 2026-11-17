"""Built-in sample meetings so the whole product can be explored without audio, GPUs or API keys."""
from __future__ import annotations

from typing import List, Tuple

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import models as m
from app.services.pipeline import _replace_transcript
from app.services.transcription import RawSegment

DEMO_MEETINGS: List[Tuple[str, List[Tuple[str, str]]]] = [
    ("Product Launch Sync — Week 1", [
        ("Priya", "Good morning everyone. The goal of today's meeting is to align on the launch plan for the mobile app and agree on who owns what."),
        ("Priya", "Marcus, can you give us a quick status on the backend?"),
        ("Marcus", "Sure. The API is about eighty percent done. The main risk is the payment integration, because the vendor sandbox keeps timing out and we are blocked on their support team."),
        ("Elena", "On the design side, the onboarding screens are finished. The only remaining piece is the dark mode variant."),
        ("Priya", "Great. So should we launch on the fifteenth or push to the end of the month?"),
        ("Sam", "From a marketing perspective the fifteenth is too tight. We need two weeks of teaser content before launch."),
        ("Marcus", "I agree. If payments are still blocked, the fifteenth is not realistic anyway."),
        ("Priya", "Okay, we've decided to move the launch date to the thirtieth of the month."),
        ("Priya", "We will use Stripe for payments instead of the current vendor, since their sandbox is far more reliable."),
        ("Marcus", "That means I need to rewrite the payment module. I'll have the new Stripe integration ready by next Wednesday."),
        ("Elena", "I'll finish the dark mode designs by Friday and share them in the team folder."),
        ("Sam", "I will draft the teaser campaign and send it to Priya for review by November 20th."),
        ("Priya", "Elena, can you also confirm the app store screenshots with the legal team?"),
        ("Sam", "One concern is the app store review process, which could delay the launch by several days if they reject the first build."),
        ("Priya", "Who will handle customer support on launch day is still undecided."),
        ("Marcus", "Let's go with Slack for daily launch updates and keep the weekly sync on Mondays."),
        ("Priya", "Perfect. Thanks everyone, that was a productive meeting."),
    ]),
    ("Launch Readiness Check — Week 2", [
        ("Priya", "Welcome back. Today we review launch readiness ahead of the thirtieth."),
        ("Marcus", "The Stripe integration is complete and passing all tests in the sandbox. Payments are no longer a blocker."),
        ("Elena", "The dark mode designs are shared, and legal approved the app store screenshots."),
        ("Sam", "The teaser campaign is live. Early signups are around twelve hundred."),
        ("Priya", "Do we have an answer on customer support coverage for launch day?"),
        ("Sam", "Not yet. I will ask Jordan from the support team to confirm staffing by Tuesday."),
        ("Elena", "The biggest risk now is the app store review, which might slip the launch if they reject our first build."),
        ("Marcus", "We decided to submit the iOS build to the app store by Thursday to leave a buffer for review."),
        ("Priya", "Sam will send the final press kit to the media list by next Monday."),
        ("Priya", "We've agreed to keep the thirtieth as the final launch date."),
    ]),
]


def _timed_segments(lines: List[Tuple[str, str]]) -> List[RawSegment]:
    out, t = [], 4.0
    for speaker, text in lines:
        dur = max(3.0, len(text.split()) / 2.6)
        out.append(RawSegment(start=round(t, 2), end=round(t + dur, 2), text=text, confidence=0.97, speaker=speaker))
        t += dur + 1.2
    return out


def create_demo_meetings(db: Session, user: m.User) -> List[m.Meeting]:
    """Create (or return the existing) demo meetings for ``user``. Processing is queued by the caller."""
    created: List[m.Meeting] = []
    for title, lines in DEMO_MEETINGS:
        existing = db.scalar(select(m.Meeting).where(m.Meeting.user_id == user.id, m.Meeting.title == title,
                                                     m.Meeting.source == "demo"))
        if existing:
            created.append(existing)
            continue
        segs = _timed_segments(lines)
        meeting = m.Meeting(user_id=user.id, title=title, filename=None, source="demo", language="en",
                            duration=round(segs[-1].end, 2), status=m.MeetingStatus.UPLOADED)
        db.add(meeting)
        db.flush()
        _replace_transcript(db, meeting, segs)
        for sp in db.scalars(select(m.Speaker).where(m.Speaker.meeting_id == meeting.id)):
            sp.display_name = sp.speaker_label  # demo speakers are named
        db.commit()
        db.refresh(meeting)
        created.append(meeting)
    return created
