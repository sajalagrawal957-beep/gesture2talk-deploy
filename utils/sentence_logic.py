class SentenceBuilder:
    def __init__(self, hold_frames=15, confidence_threshold=0.75):
        self.sentence = ""
        self.current_letter = ""
        self.frame_count = 0
        self.hold_frames = hold_frames
        self.threshold = confidence_threshold

    def update(self, letter, confidence):
        if confidence < self.threshold:
            self._reset_current()
            return self.sentence

        if letter == self.current_letter:
            self.frame_count += 1
        else:
            self.current_letter = letter
            self.frame_count = 1

        if self.frame_count == self.hold_frames:
            if letter == "SPACE":
                self.sentence += " "
            else:
                self.sentence += letter

        return self.sentence

    def backspace(self):
        self.sentence = self.sentence[:-1]
        self._reset_current()
        return self.sentence

    def clear(self):
        self.sentence = ""
        self._reset_current()
        return self.sentence

    def get_sentence(self):
        return self.sentence

    def _reset_current(self):
        self.current_letter = ""
        self.frame_count = 0