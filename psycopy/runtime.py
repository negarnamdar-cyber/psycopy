"""PsychoPy UI runtime primitives with improved visual design."""

from __future__ import annotations

from psycopy.types import TaskState


class UserAbort(Exception):
    """Raised when user aborts the experiment with ESC key."""

    pass


class PsychoPyUI:
    """Enhanced PsychoPy UI with better visual design and accessibility."""

    SHUTDOWN_CODE = "12345"

    # Modern color palette - WCAG AA compliant
    # GO: Fresh teal/green for "go" signal
    GO_COLOR = (-0.7, 0.8, -0.3)  # Teal-green
    GO_BG_COLOR = (-0.85, 0.5, -0.5)  # Darker teal background

    # STOP: Muted coral/red for "stop" signal (less aggressive than pure red)
    STOP_COLOR = (0.7, -0.6, -0.5)  # Coral
    STOP_BG_COLOR = (0.3, -0.7, -0.7)  # Darker red background

    # Neutral colors
    NEUTRAL_COLOR = (0.9, 0.9, 0.9)  # Off-white
    NEUTRAL_DIM = (0.5, 0.5, 0.5)  # Gray

    # Background: Dark gray (easier on eyes than pure black)
    BG_COLOR = (-0.6, -0.6, -0.6)

    # Instruction text
    INSTRUCTION_COLOR = (0.8, 0.8, 0.8)

    # Warning/special states
    WARNING_COLOR = (0.9, 0.5, -0.2)  # Amber
    SUCCESS_COLOR = (-0.5, 0.7, -0.4)  # Success green

    def __init__(self, fullscreen: bool):
        from psychopy import core, event, visual

        self.core = core
        self.event = event
        self.visual = visual
        self.exp_clock = core.Clock()

        self.win = visual.Window(
            size=[1920, 1080],
            fullscr=fullscreen,
            screen=0,
            allowGUI=False,
            monitor="testMonitor",
            color=self.BG_COLOR,
            colorSpace="rgb",
            blendMode="avg",
            useFBO=True,
            units="height",
        )
        self._create_visuals()

    def _create_visuals(self) -> None:
        # Fixation cross - larger and more visible
        self.fixation = self.visual.TextStim(
            win=self.win,
            text="+",
            font="Arial",
            pos=(0, 0),
            height=0.12,
            color=self.NEUTRAL_COLOR,
            colorSpace="rgb",
            bold=True,
        )

        # Sentence text - larger and better positioned
        self.sentence_text = self.visual.TextStim(
            win=self.win,
            text="",
            font="Arial",
            pos=(0, 0.25),
            height=0.06,
            wrapWidth=1.6,
            color=self.NEUTRAL_COLOR,
            colorSpace="rgb",
            alignText="center",
        )

        # State indicator (GO/STOP text) - more prominent
        self.state_indicator = self.visual.TextStim(
            win=self.win,
            text="GO",
            font="Arial",
            pos=(0, -0.25),
            height=0.18,
            color=self.GO_COLOR,
            colorSpace="rgb",
            bold=True,
            alignText="center",
        )

        # State background - larger size with subtle effect
        self.state_background = self.visual.Rect(
            win=self.win,
            width=0.7,
            height=0.3,
            pos=(0, -0.25),
            fillColor=self.GO_BG_COLOR,
            lineColor=self.GO_BG_COLOR,
            opacity=0.25,
        )

        # Progress bar for segment duration (hidden by default)
        self.progress_bar_bg = self.visual.Rect(
            win=self.win,
            width=0.7,
            height=0.03,
            pos=(0, -0.45),
            fillColor=self.NEUTRAL_DIM,
            lineColor=None,
            opacity=0.5,
        )
        self.progress_bar = self.visual.Rect(
            win=self.win,
            width=0,
            height=0.03,
            pos=(-0.35, -0.45),
            fillColor=self.NEUTRAL_COLOR,
            lineColor=None,
        )

        # Instruction text - improved readability
        self.instruction_text = self.visual.TextStim(
            win=self.win,
            text="",
            font="Arial",
            pos=(0, 0),
            height=0.05,
            wrapWidth=1.6,
            color=self.INSTRUCTION_COLOR,
            colorSpace="rgb",
            alignText="center",
        )

        # Pain warning - clearer and less alarming
        self.pain_warning = self.visual.TextStim(
            win=self.win,
            text="PAIN BLOCK NEXT\n\nExperimenter: please prepare the pain device now.\n\nPress SPACE when ready to continue.",
            font="Arial",
            pos=(0, 0),
            height=0.055,
            wrapWidth=1.6,
            color=self.WARNING_COLOR,
            colorSpace="rgb",
            bold=True,
            alignText="center",
        )

        # Help text for footer
        self.help_text = self.visual.TextStim(
            win=self.win,
            text="Press Q for coded shutdown",
            font="Arial",
            pos=(0, -0.48),
            height=0.025,
            color=self.NEUTRAL_DIM,
            colorSpace="rgb",
            alignText="center",
        )

    def _check_escape(self) -> None:
        """Check for shutdown code (Q + 12345). ESC is ignored — use Q + 12345 for graceful stop."""
        keys = self.event.getKeys(keyList=["q"])
        if "q" in keys and self._confirm_shutdown_code():
            raise UserAbort()

    def _confirm_shutdown_code(self) -> bool:
        """Show shutdown code entry screen. Only Q + 12345 works — ESC does not cancel."""
        entered = ""
        self.event.clearEvents(eventType="keyboard")
        while True:
            self.instruction_text.text = (
                "Shutdown requested\n\n"
                f"Enter code: {entered}\n\n"
                f"Type {self.SHUTDOWN_CODE} to stop gracefully.\n"
                "Press BACKSPACE to delete."
            )
            self.instruction_text.draw()
            self.help_text.draw()
            self.win.flip()

            keys = self.event.getKeys()
            for key in keys:
                if key in {"return", "num_enter", "enter"}:
                    if entered == self.SHUTDOWN_CODE:
                        self.event.clearEvents(eventType="keyboard")
                        return True
                    entered = ""
                    continue
                if key == "backspace":
                    entered = entered[:-1]
                    continue
                if key in {"1", "2", "3", "4", "5", "6", "7", "8", "9", "0"}:
                    entered += key
                    if entered == self.SHUTDOWN_CODE:
                        self.event.clearEvents(eventType="keyboard")
                        return True
                    if not self.SHUTDOWN_CODE.startswith(entered):
                        entered = ""

            self.core.wait(0.016)

    def check_spacebar(self) -> bool:
        """Non-blocking check if spacebar is pressed. Returns True if pressed, False otherwise."""
        keys = self.event.getKeys(keyList=["space"])
        return bool(keys)

    def wait_for_space(self) -> None:
        self.event.clearEvents(eventType="keyboard")
        while True:
            self._check_escape()
            if self.event.getKeys(keyList=["space"]):
                return
            self.core.wait(0.016)

    def wait(self, duration: float) -> None:
        """Wait while continuing to poll for abort keys."""
        if duration <= 0:
            return
        start = self.exp_clock.getTime()
        while (self.exp_clock.getTime() - start) < duration:
            self._check_escape()
            self.core.wait(0.016)

    def show_instructions(self, go_segmentation_enabled: bool, medoc_enabled: bool = False) -> None:
        """Show main instructions with improved formatting.

        Args:
            go_segmentation_enabled: Whether GO/STOP segmentation is enabled.
            medoc_enabled: Whether Medoc thermal stimulation is enabled.
        """
        if go_segmentation_enabled:
            if medoc_enabled:
                instructions = (
                    "EXPERIMENT INSTRUCTIONS\n\n"
                    "Vowel Experiment with Thermal Stimulation\n\n"
                    "Each trial lasts 4 minutes.\n"
                    "The screen will alternate between RED (STOP) and GREEN (GO).\n\n"
                    "IMPORTANT:\n"
                    "  • Speak 'Ahh' ONLY when the screen is GREEN\n"
                    "  • STOP immediately when the screen turns RED\n"
                    "  • Stay silent during STOP periods\n"
                    "  • The screen will switch multiple times per trial\n\n"
                    "Thermal stimulation will be applied during the experiment.\n"
                    "You may experience changes in pain/discomfort levels.\n\n"
                    "There will be 4 trials, with a 1-minute break between each.\n\n"
                    "Press SPACE to begin."
                )
            else:
                instructions = (
                    "EXPERIMENT INSTRUCTIONS\n\n"
                    "Each trial lasts 4 minutes.\n"
                    "The screen will alternate between RED (STOP) and GREEN (GO).\n\n"
                    "IMPORTANT:\n"
                    "  • Speak 'Ahh' ONLY when the screen is GREEN\n"
                    "  • STOP immediately when the screen turns RED\n"
                    "  • Stay silent during STOP periods\n"
                    "  • The screen will switch multiple times per trial\n\n"
                    "There will be 4 trials, with a 1-minute break between each.\n\n"
                    "Press SPACE to begin."
                )
        else:
            if medoc_enabled:
                instructions = (
                    "EXPERIMENT INSTRUCTIONS\n\n"
                    "Vowel Experiment with Thermal Stimulation\n\n"
                    "Each trial lasts 4 minutes.\n"
                    "The screen will stay GREEN throughout each trial.\n"
                    "Speak 'Ahh' continuously.\n\n"
                    "Thermal stimulation will be applied during the experiment.\n"
                    "You may experience changes in pain/discomfort levels.\n\n"
                    "There will be 4 trials, with a 1-minute break between each.\n\n"
                    "Press SPACE to begin."
                )
            else:
                instructions = (
                    "EXPERIMENT INSTRUCTIONS\n\n"
                    "Each trial lasts 4 minutes.\n"
                    "The screen will stay GREEN throughout each trial.\n"
                    "Speak 'Ahh' continuously.\n\n"
                    "There will be 4 trials, with a 1-minute break between each.\n\n"
                    "Press SPACE to begin."
                )
        self.instruction_text.text = instructions
        self.instruction_text.draw()
        self.help_text.draw()
        self.win.flip()
        self.wait_for_space()

    def show_speech_instructions(self, medoc_enabled: bool = False) -> None:
        """Show structured speech Q&A instructions."""
        instructions = (
            "SPEECH Q&A TASK\n\n"
            "A question will appear on screen.\n"
            "When the screen turns GREEN, you have 17–23 seconds to answer.\n"
            "Stop speaking when the screen turns RED.\n\n"
        )
        if medoc_enabled:
            instructions += (
                "Thermal stimulation will be applied during the task.\n"
                "You may experience changes in pain/discomfort levels.\n\n"
            )
        instructions += "Press SPACE to begin."

        self.instruction_text.text = instructions
        self.instruction_text.draw()
        self.help_text.draw()
        self.win.flip()
        self.wait_for_space()

    def show_pain_warning(self) -> None:
        self.pain_warning.draw()
        self.win.flip()
        self.wait_for_space()

    def show_fixation(self, duration: float = 0.5) -> None:
        self.fixation.draw()
        self.win.flip()
        start = self.exp_clock.getTime()
        while (self.exp_clock.getTime() - start) < duration:
            self._check_escape()
            self.core.wait(0.016)

    def show_progress(self, fraction: float) -> None:
        """Update trial progress bar (0.0 to 1.0)."""
        fraction = max(0.0, min(1.0, fraction))
        self.progress_bar.width = self.progress_bar_bg.width * fraction
        left_edge = self.progress_bar_bg.pos[0] - self.progress_bar_bg.width / 2
        self.progress_bar.pos = (
            left_edge + self.progress_bar.width / 2,
            self.progress_bar_bg.pos[1],
        )

    def hide_progress(self) -> None:
        """Hide the progress bar."""
        self.progress_bar.width = 0

    def apply_state(self, state: TaskState) -> None:
        if state is TaskState.GO:
            self.state_indicator.text = TaskState.GO.value
            self.state_indicator.color = self.GO_COLOR
            self.state_background.fillColor = self.GO_COLOR
            self.state_background.lineColor = self.GO_COLOR
        else:
            self.state_indicator.text = TaskState.STOP.value
            self.state_indicator.color = self.STOP_COLOR
            self.state_background.fillColor = self.STOP_COLOR
            self.state_background.lineColor = self.STOP_COLOR

    def run_segment(self, sentence: str, state: TaskState, duration: float) -> float:
        self.apply_state(state)
        self.sentence_text.text = sentence
        self.sentence_text.draw()
        self.state_background.draw()
        self.state_indicator.draw()
        before = self.exp_clock.getTime()
        self.win.flip()
        after = self.exp_clock.getTime()
        drift_ms = (after - before) * 1000.0

        segment_start = self.exp_clock.getTime()
        while (self.exp_clock.getTime() - segment_start) < duration:
            self._check_escape()
            self.core.wait(0.016)
        return drift_ms

    def run_segment_with_termination(
        self, sentence: str, state: TaskState, duration: float
    ) -> tuple[bool, float]:
        self.apply_state(state)
        self.sentence_text.text = sentence
        self.sentence_text.draw()
        self.state_background.draw()
        self.state_indicator.draw()
        before = self.exp_clock.getTime()
        self.win.flip()
        after = self.exp_clock.getTime()
        drift_ms = (after - before) * 1000.0

        segment_start = self.exp_clock.getTime()
        completed = True
        while (self.exp_clock.getTime() - segment_start) < duration:
            self._check_escape()
            if self.check_spacebar():
                completed = False
                break
            self.core.wait(0.016)
        actual_duration = (self.exp_clock.getTime() - segment_start) * 1000.0
        return completed, actual_duration

    def show_vowel_instructions(self) -> None:
        """Show vowel task instructions with improved formatting."""
        instructions = (
            "VOWEL TASK\n\n"
            "You will sustain a vowel sound ('Ahh').\n\n"
            "  • When GREEN (GO): Say 'Ahh' and hold\n"
            "  • When RED (STOP): Stop immediately\n\n"
            "Press SPACE to begin."
        )
        self.instruction_text.text = instructions
        self.instruction_text.draw()
        self.help_text.draw()
        self.win.flip()
        self.wait_for_space()

    def show_question(self, question: str) -> None:
        """Display a question on screen before the speaking period."""
        self.sentence_text.text = question
        self.sentence_text.pos = (0, 0.1)
        self.sentence_text.height = 0.06
        self.sentence_text.draw()
        self.help_text.text = "Get ready to answer..."
        self.help_text.draw()
        self.win.flip()

    def show_speech_screen(self, countdown_sec: float | None = None) -> None:
        """Show the speech screen with optional countdown timer.

        Args:
            countdown_sec: If provided, shows a countdown timer on screen.
        """
        if countdown_sec is not None:
            minutes = int(countdown_sec) // 60
            seconds = int(countdown_sec) % 60
            self.sentence_text.text = f"SPEAK\n\n{minutes:01d}:{seconds:02d}"
        else:
            self.sentence_text.text = "SPEAK"
        self.sentence_text.pos = (0, 0.05)
        self.sentence_text.height = 0.08
        self.sentence_text.draw()
        self.help_text.text = "Press Q + 12345 to stop"
        self.help_text.draw()
        self.win.flip()

    def run_question_segment(
        self, question: str, speak_duration: float = 20.0
    ) -> tuple[bool, float]:
        """Run a single question-answer segment: show question, then SPEAK for duration.

        Args:
            question: The question text to display.
            speak_duration: Number of seconds the participant has to answer.

        Returns:
            (completed, actual_duration_ms): completed is False if spacebar
            was pressed early (or shutdown requested), True otherwise.
        """
        # Show question for 3 seconds
        self.show_question(question)
        self.wait(3.0)

        # Show SPEAK with countdown for speak_duration
        self.apply_state(TaskState.GO)
        self.sentence_text.text = f"SPEAK\n\n{int(speak_duration):02d}"
        self.sentence_text.pos = (0, 0.05)
        self.sentence_text.height = 0.08
        self.sentence_text.draw()
        self.state_background.draw()
        self.state_indicator.draw()
        before = self.exp_clock.getTime()
        self.win.flip()
        after = self.exp_clock.getTime()
        drift_ms = (after - before) * 1000.0

        segment_start = self.exp_clock.getTime()
        completed = True
        last_displayed_second: int | None = None
        while (self.exp_clock.getTime() - segment_start) < speak_duration:
            self._check_escape()
            if self.check_spacebar():
                completed = False
                break
            remaining = speak_duration - (self.exp_clock.getTime() - segment_start)
            displayed_second = int(remaining)
            if displayed_second != last_displayed_second:
                self.show_speech_screen(countdown_sec=remaining)
                last_displayed_second = displayed_second
            self.core.wait(0.1)

        actual_duration = (self.exp_clock.getTime() - segment_start) * 1000.0
        return completed, actual_duration

    def show_rt_instructions(self) -> None:
        """Show reaction time task instructions with improved formatting."""
        instructions = (
            "REACTION TIME TASK\n\n"
            "A fixation cross (+) will appear.\n\n"
            "  • Wait for it to turn GREEN\n"
            "  • When you see 'PRESS!' — hit SPACE fast!\n\n"
            "Press SPACE to begin."
        )
        self.instruction_text.text = instructions
        self.instruction_text.draw()
        self.help_text.draw()
        self.win.flip()
        self.wait_for_space()

    def run_rt_trial(
        self, rng, jitter_min: float = 1.0, jitter_max: float = 3.0, timeout: float = 2.0
    ) -> tuple[float, bool]:
        """Display fixation with random jitter, then a GO cue. Returns (rt_ms, missed)."""
        # Fixation period with jitter
        self.fixation.color = self.NEUTRAL_COLOR
        self.fixation.draw()
        self.win.flip()
        jitter = rng.uniform(jitter_min, jitter_max)
        jitter_start = self.exp_clock.getTime()
        while (self.exp_clock.getTime() - jitter_start) < jitter:
            self._check_escape()
            self.core.wait(0.001)

        # GO cue
        self.event.clearEvents(eventType="keyboard")
        self.fixation.color = self.GO_COLOR
        self.state_indicator.text = "PRESS!"
        self.state_indicator.color = self.GO_COLOR
        self.state_background.fillColor = self.GO_COLOR
        self.state_background.lineColor = self.GO_COLOR
        self.fixation.draw()
        self.state_background.draw()
        self.state_indicator.draw()
        self.win.flip()
        cue_onset = self.exp_clock.getTime()

        # Wait for spacebar response
        rt_ms = -1.0
        missed = True
        response_clock = self.core.Clock()
        response_clock.reset()
        while response_clock.getTime() < timeout:
            self._check_escape()
            keys = self.event.getKeys(keyList=["space"])
            if keys:
                rt_ms = response_clock.getTime() * 1000.0
                missed = False
                break
            self.core.wait(0.001)

        # Reset display
        self.fixation.color = self.NEUTRAL_COLOR
        self.win.flip()
        self.core.wait(0.5)
        return rt_ms, missed

    def show_set_waiting_screen(self, set_num: int) -> None:
        """Display waiting screen between sets, blocking for spacebar.

        Args:
            set_num: Set number (0-indexed).

        Displays: "Set {set_num+1}/8 complete. Press SPACE when ready to continue."
        Clears screen after key press.
        """
        message = f"Set {set_num + 1}/8 complete.\n\nPress SPACE when ready to continue."
        self.instruction_text.text = message
        self.instruction_text.draw()
        self.help_text.draw()
        self.win.flip()
        self.event.clearEvents(eventType="keyboard")
        while True:
            self._check_escape()
            if self.event.getKeys(keyList=["space"]):
                break
            self.core.wait(0.016)
        self.win.flip()

    def show_completion(self) -> None:
        """Show experiment completion message with improved formatting."""
        self.instruction_text.text = (
            "EXPERIMENT COMPLETE\n\nThank you for participating!\n\n"
            "Thermal stimulation data has been logged.\n\n"
            "Press SPACE to exit."
        )
        self.instruction_text.draw()
        self.win.flip()
        while True:
            self._check_escape()
            if self.event.getKeys(keyList=["space"]):
                return
            self.core.wait(0.016)

    def close(self) -> None:
        self.win.close()
        self.core.quit()
