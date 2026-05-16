import plotly.graph_objects as go

from mandarin_speech_coach.tones.tone_colors import (
    TONE_COLORS,
)


class PronunciationPlotBuilder:
    def build(
        self,
        alignment,
        pitch_track,
        target_tones,
    ):
        fig = go.Figure()

        # Add pitch contour
        fig.add_trace(
            go.Scatter(
                x=pitch_track.times,
                y=pitch_track.frequencies,
                mode="lines",
                line=dict(color="red", width=2),
                name="Pitch (F0)",  # TODO What is F0 here?
            )
        )

        # Add aligned segments and tone colors
        for i, seg in enumerate(alignment.segments):
            # Get tone color
            tone_char = target_tones[i][0][-1]
            color = TONE_COLORS.get(tone_char, TONE_COLORS["5"])

            # Add colored background strip
            fig.add_shape(
                type="rect",
                x0=seg.start,
                y0=0,
                x1=seg.end,
                y1=1,
                xref="x",
                yref="paper",
                fillcolor=color,
                layer="below",
                line_width=0,
            )

            # Add character annotation at the top
            fig.add_annotation(
                x=(seg.start + seg.end) / 2,
                y=0.95,
                yref="paper",
                text=f"<b>{seg.label}</b>",
                showarrow=False,
                font=dict(size=20, color="black"),
            )

        # Update layout for a clean look
        fig.update_layout(
            title="Pronunciation Analysis: Pitch Contour and Alignment",
            xaxis_title="Time (s)",
            yaxis_title="Pitch (Hz)",
            showlegend=False,
            xaxis=dict(range=[0, pitch_track.duration]),
            yaxis=dict(rangemode="tozero"),  # Ensure y-axis starts at 0
            height=400,
        )

        return fig
