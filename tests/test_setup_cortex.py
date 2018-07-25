import pytest
from cortex._lib import setup_cortex
from cortex.built_ins.models.classifier import ImageClassification

@pytest.mark.run_these_please
def test_setup_cortex():
    classifier = ImageClassification()
    args = setup_cortex(model=classifier)
