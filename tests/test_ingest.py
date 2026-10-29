import unittest

from src.ingest import split_into_chunks


class TestSplitIntoChunks(unittest.TestCase):

    def test_empty_text_returns_no_chunks(self):
        self.assertEqual(split_into_chunks(""), [])

    def test_single_paragraph_returns_chunk(self):
        text = "Python makes automation easier."
        self.assertEqual(
            split_into_chunks(text),
            ["Python makes automation easier."]
        )

    def test_topic_is_attached_to_following_paragraph(self):
        text = (
            "Topic: Linux\n\n"
            "The df command reports filesystem capacity."
        )
        chunks = split_into_chunks(text)

        self.assertEqual(len(chunks), 1)
        self.assertIn("Topic: Linux.", chunks[0])
        self.assertIn("df command", chunks[0])

    def test_long_paragraph_is_split(self):
        text = " ".join(["word"] * 205)
        chunks = split_into_chunks(text, max_words=100)

        self.assertEqual(len(chunks), 3)
        self.assertEqual(
            [len(chunk.split()) for chunk in chunks],
            [100, 100, 5]
        )

    def test_blank_paragraphs_are_ignored(self):
        text = "\n\nPython is useful.\n\n"
        self.assertEqual(split_into_chunks(text), ["Python is useful."])


if __name__ == "__main__":
    unittest.main()
