"""Modules containing all request/response schemas."""
from marshmallow import Schema, fields, validate


class TweetUrlSchema(Schema):
    """Request schema for predict API."""

    tweetUrl = fields.String()

class ResultSchema(Schema):
    """Response schema for predict API."""

    clip = fields.String()
    mfas = fields.String()