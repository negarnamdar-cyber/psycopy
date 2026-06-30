"""Short on-screen practice demo of the STOP/GO task mechanics.

Runs a brief demonstration of both the vowel and speech tasks so
participants can see what to do before the real experiment. It does NOT
start the Medoc device or record audio -- it only drives the PsychoPy
window to show the STOP/GO cues.
"""

from __future__ import annotations

import logging

from psycopy.runtime import PsychoPyUI, UserAbort
from psycopy.types import TaskState

logger = logging.getLogger("psycopy.practice_demo")

# --- Vowel demo timing (seconds) -------------------------------------------
VOWEL_STOP_SEC = 3.0
VOWEL_GO_SEC = 5.0
VOWEL_GO_SEGMENTS = 2  # number of GO periods (STOP, GO, STOP, GO, STOP)

# --- Speech demo timing (seconds) -------------------------------------------
SPEECH_READ_SEC = 4.0
SPEECH_RATE_SEC = 3.0
SPEECH_ANSWER_SEC = 6.0

# Short sample questions so they can be read quickly during the demo.
DEMO_QUESTIONS = (
    "How would you describe your pain right now?",
    "Where in your body do you feel it most?",
)

VOWEL_TEXT = "Ahh"


class PracticeDemo:
    """Brief on-screen demo of STOP/GO mechanics for both tasks.

    No Medoc, no audio -- just the visual cues.
    """

    def __init__(self, config) -> None:
        self.config = config
        self.ui = PsychoPyUI(fullscreen=config.fullscreen)

    def _show_text(self, text: str, footer: str = "Press SPACE to continue") -> None:
        self.ui.help_text.text = footer
        self.ui.instruction_text.text = text
        self.ui.instruction_text.draw()
        self.ui.help_text.draw()
        self.ui.win.flip()
        self.ui.wait_for_space()

    def _display_state(self, state: TaskState, text: str) -> None:
        self.ui.apply_state(state)
        self.ui.sentence_text.text = text
        self.ui.sentence_text.draw()
        self.ui.state_background.draw()
        self.ui.state_indicator.draw()
        self.ui.win.flip()

    def _run_vowel_demo(self) -> None:
        for _ in range(VOWEL_GO_SEGMENTS):
            self._display_state(TaskState.STOP, VOWEL_TEXT)
            self.ui.wait(VOWEL_STOP_SEC)
            self._display_state(TaskState.GO, VOWEL_TEXT)
            self.ui.wait(VOWEL_GO_SEC)
        # Final STOP (rest) period.
        self._display_state(TaskState.STOP, VOWEL_TEXT)
        self.ui.wait(VOWEL_STOP_SEC)

    def _run_speech_demo(self) -> None:
        for question in DEMO_QUESTIONS:
            # STOP: read the question.
            self._display_state(TaskState.STOP, question)
            self.ui.wait(SPEECH_READ_SEC)
            # GO: answer aloud.
            self._display_state(TaskState.GO, question)
            self.ui.wait(SPEECH_ANSWER_SEC)
            # STOP: "Rate your pain" prompt (after speaking).
            self.ui.show_rate_pain_prompt()
            self.ui.wait(SPEECH_RATE_SEC)
        # Final brief STOP (stop speaking).
        self._display_state(TaskState.STOP, "Stop")
        self.ui.wait(SPEECH_RATE_SEC)

    def run(self) -> None:
        try:
            self._show_text(
                "PRACTICE MODE -- VOWEL TASK\n\n"
                "This is a short demo. No thermal stimulation and no audio will "
                'be recorded -- it only shows the STOP/GO cues.\n\n'
                "The screen alternates RED (STOP) and GREEN (GO):\n"
                '  - GREEN (GO): say "Ahh" and hold\n'
                "  - RED (STOP): stop speaking immediately\n\n"
                "Press SPACE to begin."
            )
            self._run_vowel_demo()

            self._show_text(
                "Vowel practice complete!\n\n"
                "PRACTICE MODE -- SPEECH Q&A TASK\n\n"
                "  - A question appears on RED -- read it silently\n"
                "  - When the screen turns GREEN -- answer aloud\n"
                '  - A "Rate your pain" prompt follows (turns RED)\n'
                "  - Stop speaking when the screen turns RED\n\n"
                "Press SPACE to begin."
            )
            self._run_speech_demo()

            self._show_text(
                "PRACTICE COMPLETE\n\n"
                "That's how both tasks work.\n"
                "Remember: GREEN = speak, RED = stop.\n\n"
                "You're ready for the real experiment.\n\n"
                "Press SPACE to exit."
            )
        except UserAbort:
            logger.info("Practice demo interrupted by user.")
        finally:
            self.ui.close()
