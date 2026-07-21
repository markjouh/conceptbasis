import base64
import io

import pytest
from PIL import Image

from conceptbasis.site import public_nav, thumbnail_data_url


def test_public_nav_marks_exactly_one_page_active():
    navigation = public_nav("dictionary")
    assert navigation.count('class="here"') == 1
    assert '<a href="dictionary.html" class="here">' in navigation
    with pytest.raises(ValueError, match="unknown public page"):
        public_nav("missing")


def test_public_thumbnail_is_bounded_webp(tmp_path):
    source = tmp_path / "source.png"
    Image.new("RGB", (600, 300), (10, 20, 30)).save(source)
    encoded = thumbnail_data_url(source, size=120)
    assert encoded.startswith("data:image/webp;base64,")
    with Image.open(io.BytesIO(base64.b64decode(encoded.partition(",")[2]))) as image:
        assert image.format == "WEBP"
        assert image.size == (120, 60)
