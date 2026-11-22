"""Global CSS: deep navy / electric blue theme."""
NAVY, BG, CARD, BLUE, ACCENT = '#0B1A2E', '#F0F4FA', '#FFFFFF', '#1D4ED8', '#3B82F6'
MUTED, OK, WARN, ERR = '#64748B', '#16A34A', '#D97706', '#DC2626'

CSS = f"""
<style>
.stApp {{ background: {BG}; }}
[data-testid="stSidebar"] {{ background: {NAVY}; }}
[data-testid="stSidebar"] * {{ color: #E2E8F0 !important; }}
[data-testid="stSidebar"] hr {{ border-color: #1E3A5F; }}
.block-container {{ padding-top: 3.5rem; max-width: 1400px; }}
.app-header {{ background: {NAVY}; color: #fff; padding: 16px 22px; border-radius: 10px; margin-bottom: 18px;
  display: flex; justify-content: space-between; align-items: center; border-left: 6px solid {ACCENT}; }}
.app-header h1 {{ margin: 0; font-size: 1.45rem; color: #fff; padding: 0; }}
.app-header .sub {{ color: #93C5FD; font-size: .85rem; }}
.app-header .right {{ text-align: right; font-size: .8rem; color: #CBD5E1; }}
.card {{ background: {CARD}; border-radius: 10px; padding: 14px 16px; box-shadow: 0 1px 3px rgba(11,26,46,.10);
  border-top: 3px solid {ACCENT}; height: 100%; }}
.card .label {{ color: {MUTED}; font-size: .75rem; text-transform: uppercase; letter-spacing: .04em; }}
.card .value {{ color: {NAVY}; font-size: 1.7rem; font-weight: 700; line-height: 1.2; }}
.card .note {{ color: {MUTED}; font-size: .75rem; }}
.card.ok {{ border-top-color: {OK}; }} .card.warn {{ border-top-color: {WARN}; }} .card.err {{ border-top-color: {ERR}; }}
.badge {{ display: inline-block; padding: 2px 9px; border-radius: 999px; font-size: .75rem; font-weight: 600; }}
.b-ok {{ background: #DCFCE7; color: #166534; }} .b-warn {{ background: #FEF3C7; color: #92400E; }}
.b-err {{ background: #FEE2E2; color: #991B1B; }} .b-info {{ background: #DBEAFE; color: #1E40AF; }}
.b-mute {{ background: #E2E8F0; color: #475569; }}
.console {{ background: #0A1220; color: #CBD5E1; font-family: ui-monospace, Menlo, Consolas, monospace; font-size: .78rem;
  padding: 12px 14px; border-radius: 8px; max-height: 340px; overflow-y: auto; white-space: pre-wrap; }}
.section-title {{ color: {NAVY}; font-weight: 700; font-size: 1.05rem; margin: 6px 0 8px; }}
.empty {{ background: {CARD}; border: 1px dashed #94A3B8; border-radius: 10px; padding: 26px; text-align: center; color: {MUTED}; }}
div[data-testid="stTabs"] button[aria-selected="true"] {{ color: {BLUE}; }}
.stButton > button[kind="primary"] {{ background: {BLUE}; border-color: {BLUE}; }}
</style>
"""
