"""
This file contains the replacement AI panel code.
It will be patched into theta_dashboard_v4_modern.py.
"""

# ============================================================
# LAYOUT SECTION — replaces everything from the old AI ANALYST
# CHAT PANEL comment block through the closing ]) of app.layout
# ============================================================

LAYOUT_REPLACEMENT = r'''
        # ══════════════════════════════════════════════════════════════════
        # AI MULTI-AGENT ANALYST — OpenRouter + Perplexity + xAI Pipeline
        # ══════════════════════════════════════════════════════════════════
        html.Div(
            style={
                "padding": "0 28px 28px 28px",
                "marginTop": "12px",
            },
            children=[
                # Section header
                html.Div(
                    style={
                        "display": "flex", "alignItems": "center", "gap": "12px",
                        "marginBottom": "14px", "paddingTop": "14px",
                        "borderTop": f"1px solid {MC['border']}",
                    },
                    children=[
                        html.Div(style={
                            "width": "4px", "height": "26px", "borderRadius": "2px",
                            "background": f"linear-gradient(180deg, #8b5cf6, {MC['accent']})",
                        }),
                        html.Div(children=[
                            html.Div("AI MULTI-AGENT ANALYST", style={
                                "fontSize": "16px", "fontWeight": 800,
                                "letterSpacing": "1.5px", "color": MC["text"],
                            }),
                            html.Div("7-Agent pipeline: Perplexity + xAI + OpenRouter | Manual Q&A below", style={
                                "fontSize": "12px", "color": MC["text_muted"],
                                "letterSpacing": "0.3px",
                            }),
                        ]),
                    ],
                ),

                # ── Three API Key Bars ──
                html.Div(
                    style={
                        "display": "flex", "gap": "12px", "flexWrap": "wrap",
                        "alignItems": "flex-end", "marginBottom": "12px",
                    },
                    children=[
                        # Perplexity API Key
                        html.Div(style={"flex": "1", "minWidth": "220px"}, children=[
                            html.Label("PERPLEXITY API KEY", style={
                                "fontSize": "11px", "fontWeight": 700, "color": MC["text_sec"],
                                "letterSpacing": "0.8px", "marginBottom": "4px",
                                "display": "block",
                            }),
                            dcc.Input(
                                id="ai-pplx-key",
                                type="password",
                                placeholder="pplx-...",
                                debounce=True,
                                style={
                                    "width": "100%", "padding": "8px 12px",
                                    "backgroundColor": MC["bg_input"],
                                    "border": f"1px solid {MC['border']}",
                                    "borderRadius": "6px", "color": MC["text"],
                                    "fontSize": "13px", "fontFamily": "'JetBrains Mono', monospace",
                                    "outline": "none",
                                },
                            ),
                        ]),
                        # xAI API Key
                        html.Div(style={"flex": "1", "minWidth": "220px"}, children=[
                            html.Label("xAI API KEY", style={
                                "fontSize": "11px", "fontWeight": 700, "color": MC["text_sec"],
                                "letterSpacing": "0.8px", "marginBottom": "4px",
                                "display": "block",
                            }),
                            dcc.Input(
                                id="ai-xai-key",
                                type="password",
                                placeholder="xai-...",
                                debounce=True,
                                style={
                                    "width": "100%", "padding": "8px 12px",
                                    "backgroundColor": MC["bg_input"],
                                    "border": f"1px solid {MC['border']}",
                                    "borderRadius": "6px", "color": MC["text"],
                                    "fontSize": "13px", "fontFamily": "'JetBrains Mono', monospace",
                                    "outline": "none",
                                },
                            ),
                        ]),
                        # OpenRouter API Key
                        html.Div(style={"flex": "1", "minWidth": "220px"}, children=[
                            html.Label("OPENROUTER API KEY", style={
                                "fontSize": "11px", "fontWeight": 700, "color": MC["text_sec"],
                                "letterSpacing": "0.8px", "marginBottom": "4px",
                                "display": "block",
                            }),
                            dcc.Input(
                                id="ai-api-key",
                                type="password",
                                placeholder="sk-or-v1-...",
                                debounce=True,
                                style={
                                    "width": "100%", "padding": "8px 12px",
                                    "backgroundColor": MC["bg_input"],
                                    "border": f"1px solid {MC['border']}",
                                    "borderRadius": "6px", "color": MC["text"],
                                    "fontSize": "13px", "fontFamily": "'JetBrains Mono', monospace",
                                    "outline": "none",
                                },
                            ),
                        ]),
                    ],
                ),

                # ── Pipeline Control Row: RUN button + Status Bar ──
                html.Div(
                    style={
                        "display": "flex", "gap": "10px", "alignItems": "center",
                        "marginBottom": "12px", "flexWrap": "wrap",
                    },
                    children=[
                        html.Button("\u25B6  RUN AI PIPELINE", id="ai-pipeline-btn", n_clicks=0, style={
                            "backgroundColor": "#8b5cf6", "color": "#fff",
                            "border": "none", "padding": "10px 24px",
                            "borderRadius": "8px", "cursor": "pointer",
                            "fontWeight": 700, "fontSize": "13px",
                            "letterSpacing": "0.5px", "whiteSpace": "nowrap",
                            "minWidth": "180px",
                        }),
                        html.Div(
                            id="ai-pipeline-status",
                            style={
                                "flex": "1", "minWidth": "300px",
                                "backgroundColor": MC["bg_input"],
                                "border": f"1px solid {MC['border']}",
                                "borderRadius": "8px",
                                "padding": "8px 14px",
                                "fontSize": "12px",
                                "fontFamily": "'JetBrains Mono', monospace",
                                "color": MC["text_muted"],
                                "minHeight": "36px",
                                "display": "flex", "alignItems": "center",
                            },
                            children="Pipeline idle — press RUN to start 7-agent analysis",
                        ),
                    ],
                ),

                # ── Chat History Display ──
                html.Div(
                    id="ai-chat-history",
                    style={
                        "backgroundColor": MC["bg_card"],
                        "border": f"1px solid {MC['border']}",
                        "borderRadius": "10px",
                        "padding": "16px",
                        "minHeight": "200px",
                        "maxHeight": "700px",
                        "overflowY": "auto",
                        "marginBottom": "10px",
                        "fontFamily": "'Inter', system-ui, sans-serif",
                        "fontSize": "13px",
                        "lineHeight": "1.6",
                    },
                    children=[
                        html.Div("Pipeline results and Q&A responses will appear here...", style={
                            "color": MC["text_muted"], "fontStyle": "italic",
                        }),
                    ],
                ),

                # ── Manual Q&A Row: Model dropdown + Question + ASK + CLEAR ──
                html.Div(
                    style={
                        "display": "flex", "gap": "8px", "alignItems": "stretch",
                        "flexWrap": "wrap",
                    },
                    children=[
                        html.Div(style={"minWidth": "200px"}, children=[
                            dcc.Dropdown(
                                id="ai-model-select",
                                options=[
                                    {"label": "GPT-5.4 (OpenRouter)",               "value": "openrouter|openai/gpt-5.4"},
                                    {"label": "Claude Sonnet 4.6 (OpenRouter)",      "value": "openrouter|anthropic/claude-sonnet-4.6"},
                                    {"label": "Claude Opus 4.6 (OpenRouter)",        "value": "openrouter|anthropic/claude-opus-4-6"},
                                    {"label": "Gemini 3.1 Pro (OpenRouter)",         "value": "openrouter|google/gemini-3.1-pro"},
                                    {"label": "DeepSeek V3.2 (OpenRouter)",          "value": "openrouter|deepseek/deepseek-v3.2"},
                                    {"label": "Grok 4.20 (OpenRouter)",              "value": "openrouter|x-ai/grok-4.20-multi-agent-beta"},
                                    {"label": "Free Router (OpenRouter)",            "value": "openrouter|openrouter/free"},
                                    {"label": "Sonar Pro (Perplexity)",              "value": "perplexity|sonar-pro"},
                                    {"label": "Sonar Reasoning (Perplexity)",        "value": "perplexity|sonar-reasoning-pro"},
                                    {"label": "Grok 4 (xAI Native)",                "value": "xai|grok-4-0709"},
                                    {"label": "Grok 4.20 (xAI Native)",             "value": "xai|grok-4.20-beta-latest-non-reasoning"},
                                ],
                                value="openrouter|openai/gpt-5.4",
                                multi=False,
                                clearable=False,
                                placeholder="Select model...",
                                className="mc-dropdown",
                                style={
                                    "backgroundColor": MC["bg_input"],
                                    "color": MC["text"],
                                    "fontSize": "13px",
                                    "fontWeight": 600,
                                    "minWidth": "200px",
                                },
                            ),
                        ]),
                        dcc.Input(
                            id="ai-question-input",
                            type="text",
                            placeholder="Ask a follow-up question about the pipeline results or any market question...",
                            debounce=False,
                            n_submit=0,
                            style={
                                "flex": "1", "padding": "10px 14px",
                                "backgroundColor": MC["bg_input"],
                                "border": f"1px solid {MC['border']}",
                                "borderRadius": "8px", "color": MC["text"],
                                "fontSize": "14px", "minWidth": "200px",
                                "fontFamily": "'Inter', system-ui, sans-serif",
                                "outline": "none",
                            },
                        ),
                        html.Button("ASK AI", id="ai-send-btn", n_clicks=0, style={
                            "backgroundColor": "#3b82f6", "color": "#fff",
                            "border": "none", "padding": "10px 22px",
                            "borderRadius": "8px", "cursor": "pointer",
                            "fontWeight": 700, "fontSize": "13px",
                            "letterSpacing": "0.5px", "whiteSpace": "nowrap",
                        }),
                        html.Button("CLEAR", id="ai-clear-btn", n_clicks=0, style={
                            "backgroundColor": "#4b5563", "color": "#fff",
                            "border": "none", "padding": "10px 14px",
                            "borderRadius": "8px", "cursor": "pointer",
                            "fontWeight": 700, "fontSize": "13px",
                            "letterSpacing": "0.5px",
                        }),
                    ],
                ),

                # Hidden stores
                dcc.Store(id="ai-chat-store", data=[]),
                dcc.Store(id="ai-pipeline-state", data={"running": False, "step": 0, "results": {}}),
                dcc.Interval(id="ai-pipeline-tick", interval=2000, n_intervals=0, disabled=True),
                # Removed web search checklist - web search is always-on for pipeline
            ],
        ),
    ]
)
'''