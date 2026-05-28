"""Improved UI components and instructions for better user experience."""

from __future__ import annotations

# Instruction text templates with better formatting and clarity

INSTRUCTIONS_MAIN = """
INSTRUCTIONS

You will read sentences aloud during this experiment.

IMPORTANT RULES:

  • Speak ONLY when the screen is GREEN and shows "GO"
  • Stop speaking immediately when the screen turns RED and shows "STOP"
  • Remain silent until the screen turns GREEN again

The screen will switch between GO and STOP multiple times during each trial.

Press SPACE to begin the experiment.
"""

INSTRUCTIONS_SEGMENTATION_OFF = """
INSTRUCTIONS

You will read sentences aloud during this experiment.

The screen will remain GREEN throughout each trial.
Speak continuously until you finish the sentence.

Press SPACE to begin the experiment.
"""

INSTRUCTIONS_VOWEL = """
VOWEL TASK INSTRUCTIONS

You will sustain a vowel sound during this task.

  • When the screen is GREEN ("GO"): Say "Ahh" and hold the sound
  • When the screen turns RED ("STOP"): Stop immediately and stay silent

Press SPACE to begin.
"""

INSTRUCTIONS_RT = """
REACTION TIME TASK INSTRUCTIONS

A fixation cross (+) will appear on screen.

  • Wait for the cross to turn GREEN
  • When you see "PRESS!" — press SPACE as fast as you can
  • Try to respond as quickly as possible

Press SPACE to begin.
"""

INSTRUCTIONS_PAIN_WARNING = """
PAIN BLOCK PREPARATION

The next block will include painful stimuli.

Experimenter: Please ensure the pain device is ready.

Press SPACE when you are ready to continue.
"""

COMPLETION_MESSAGE = """
EXPERIMENT COMPLETE

Thank you for your participation!

Your data has been saved.

Press SPACE to exit.
"""

# Styling constants for visual consistency
UI_CONSTANTS = {
    # Text sizing relative to window height
    "text_height_small": 0.035,
    "text_height_normal": 0.05,
    "text_height_large": 0.07,
    "text_height_title": 0.09,
    
    # Spacing
    "line_spacing": 0.08,
    "paragraph_spacing": 0.12,
    "section_spacing": 0.2,
    
    # Colors (RGB normalized -1 to 1)
    "color_text_primary": (0.9, 0.9, 0.9),
    "color_text_secondary": (0.6, 0.6, 0.6),
    "color_text_emphasis": (0.95, 0.95, 0.95),
    "color_go": (-0.7, 0.8, -0.3),
    "color_stop": (0.7, -0.6, -0.5),
    "color_warning": (0.9, 0.5, -0.2),
    "color_success": (-0.5, 0.7, -0.4),
    
    # Timing
    "fixation_duration_min": 0.4,
    "fixation_duration_max": 0.6,
    "iti_min": 0.4,
    "iti_max": 0.8,
}


def format_instructions(text: str, width: int = 70) -> str:
    """Format instructions with proper line wrapping.
    
    Args:
        text: Raw instruction text
        width: Maximum line width
        
    Returns:
        Formatted text ready for display
    """
    lines = text.strip().split("\n")
    formatted = []
    
    for line in lines:
        if line.strip() == "":
            formatted.append("")
        elif line.startswith("  "):
            # Preserve indented lines
            formatted.append(line)
        elif len(line) <= width:
            formatted.append(line)
        else:
            # Simple word wrap
            words = line.split()
            current = ""
            for word in words:
                if len(current) + len(word) + 1 <= width:
                    current = current + " " + word if current else word
                else:
                    formatted.append(current)
                    current = word
            if current:
                formatted.append(current)
    
    return "\n".join(formatted)


def get_instructions(segmentation_enabled: bool) -> str:
    """Get the appropriate instructions based on configuration.
    
    Args:
        segmentation_enabled: Whether GO/STOP segmentation is enabled
        
    Returns:
        Formatted instruction text
    """
    if segmentation_enabled:
        return format_instructions(INSTRUCTIONS_MAIN)
    else:
        return format_instructions(INSTRUCTIONS_SEGMENTATION_OFF)
