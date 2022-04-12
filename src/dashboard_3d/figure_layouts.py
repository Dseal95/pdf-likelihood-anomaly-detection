"""Configure trace layouts for app."""

layout_3d = dict(
    showlegend=False,
    coloraxis_showscale=False,
    plot_bgcolor="rgb(255,255,255)",
    clickmode="event",
    # Marginal X
    xaxis=dict(
        gridcolor="lightgrey",
        showgrid=True,
        title=dict(font=dict(color="black", family="Verdana"), text="Tag X"),
        showline=True,
        linecolor="black",
        linewidth=2,
        mirror=True,
        dtick=1,
    ),
    yaxis=dict(
        gridcolor="lightgrey",
        showgrid=True,
        title=dict(font=dict(color="black", family="Verdana"), text="P(x)"),
        showline=True,
        linecolor="black",
        linewidth=2,
        mirror=True,
        rangemode="tozero",
    ),
    # Scoring Function
    xaxis2=dict(
        gridcolor="lightgrey",
        showgrid=True,
        title=dict(font=dict(color="black", family="Verdana"), text="P(x,y,z)"),
        showline=True,
        linecolor="black",
        linewidth=2,
        mirror=True,
        rangemode="tozero",
        dtick=0.01,
    ),
    yaxis2=dict(
        gridcolor="lightgrey",
        range=[0, 1],
        side="right",
        title=dict(
            font=dict(color="black", family="Verdana"),
            text="Pr[L\u2093 \u2264 P(x,y,z)",
        ),
        showline=True,
        linecolor="black",
        linewidth=2,
        mirror=True,
    ),
    # Marginal Y
    xaxis3=dict(
        gridcolor="lightgrey",
        title=dict(font=dict(color="black", family="Verdana"), text="Tag Y"),
        showline=True,
        linecolor="black",
        linewidth=2,
        mirror=True,
        dtick=1,
    ),
    yaxis3=dict(
        gridcolor="lightgrey",
        title=dict(font=dict(color="black", family="Verdana"), text="P(y)"),
        showline=True,
        linecolor="black",
        linewidth=2,
        mirror=True,
        rangemode="tozero",
    ),
    # Marginal Z
    xaxis4=dict(
        gridcolor="lightgrey",
        title=dict(font=dict(color="black", family="Verdana"), text="Tag Z"),
        showline=True,
        linecolor="black",
        linewidth=2,
        mirror=True,
        dtick=1,
    ),
    yaxis4=dict(
        gridcolor="lightgrey",
        title=dict(font=dict(color="black", family="Verdana"), text="P(z)"),
        showline=True,
        linecolor="black",
        linewidth=2,
        mirror=True,
        rangemode="tozero",
    ),
    title=dict(
        text="Scoring Function",
        x=0.92,
        y=0.91,
        yanchor="top",
        font=dict(family="Verdana", size=16, color="black"),
    ),
)

scene_3d = dict(
    camera=dict(
        center=dict(x=0, y=0, z=0), eye=dict(x=2, y=-1.6, z=0.5), up=dict(x=0, y=0, z=1)
    ),
    dragmode="turntable",
    xaxis=dict(
        showgrid=True,
        backgroundcolor="rgb(255,255,255)",
        gridcolor="lightgrey",
        title=dict(text="Tag X", font=dict(family="Verdana", color="black")),
    ),
    yaxis=dict(
        showgrid=True,
        backgroundcolor="rgb(255,255,255)",
        gridcolor="lightgrey",
        title=dict(text="Tag Y", font=dict(family="Verdana", color="black")),
    ),
    zaxis=dict(
        showgrid=True,
        backgroundcolor="rgb(255,255,255)",
        gridcolor="lightgrey",
        title=dict(text="Tag Z", font=dict(family="Verdana", color="black")),
    ),
    bgcolor="rgb(255,255,255)",
)
