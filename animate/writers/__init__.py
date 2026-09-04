from .mp4_writer import Mp4Writer
from .pdf_writer import PdfWriter
from .base_writer import AnimationWriter

from dataclasses import fields
from typing import Any, TypeVar, cast

WRITERS: dict[str, type[AnimationWriter]] = {
    '.mp4' : Mp4Writer,
    # '.pdf': PdfWriter # broken :(
}

W = TypeVar("W", bound=AnimationWriter)

def make_writer(writer_cls: type[W], **kwargs: Any) -> W:
    accepted = {
        field.name
        for field in fields(cast(Any, writer_cls))
    }

    return writer_cls(**{
        key: value
        for key, value in kwargs.items()
        if key in accepted
    })
