"""Consent rules shared by every WhatsApp send path (refillcare.whatsapp.consent)."""
from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import database.models  # noqa: F401
from database.connection import Base
from refillcare.whatsapp.consent import (
    CONSENT_START,
    CONSENT_STOP,
    is_opted_out,
    load_opt_out_index,
    parse_consent_keyword,
    set_opt_in,
    set_opt_in_by_phone,
)


@pytest.fixture
def db():
    eng = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(bind=eng)
    s = sessionmaker(bind=eng)()
    yield s
    s.close()


@pytest.mark.parametrize("msg,expected", [
    ("STOP", CONSENT_STOP),
    ("Stop.", CONSENT_STOP),
    ("please stop sending", CONSENT_STOP),
    ("UNSUBSCRIBE", CONSENT_STOP),
    ("opt out", CONSENT_STOP),
    ("START", CONSENT_START),
    (" start ", CONSENT_START),
    ("STOP START", CONSENT_STOP),          # ambiguous -> the safe direction
    ("Yes, I bought it yesterday", None),  # never an opt-in
    ("restart", None),
    ("START.", None),
    ("Can I cancel this medicine?", None),
    ("", None),
])
def test_parse_consent_keyword(msg, expected):
    assert parse_consent_keyword(msg) == expected


def test_phone_opt_out_covers_every_record_with_that_number(db):
    set_opt_in(db, "C1", True, phone="9876543210")
    set_opt_in(db, "C2", True, phone="+91 98765 43210")
    assert sorted(set_opt_in_by_phone(db, "919876543210", opt_in=False)) == ["C1", "C2"]
    idx = load_opt_out_index(db)
    assert idx.blocks(phone="09876543210") and idx.blocks(customer_id="C2")
    assert not idx.blocks(customer_id="C3", phone="9000000000")


def test_unknown_phone_creates_consent_record(db):
    assert set_opt_in_by_phone(db, "919111122222", opt_in=False) == ["CUST_9111122222"]
    assert is_opted_out(db, phone="9111122222")
    set_opt_in_by_phone(db, "9111122222", opt_in=True)
    assert not is_opted_out(db, phone="9111122222")
