#!/usr/bin/env python3
"""生成 scored_result.csv 的静态 HTML + ECharts 可视化报告。

用法：
  cd myanalyser && source .venv312/bin/activate
  python tools/gen_scoreboard_html.py -i artifacts/filter_score_run_3/scored_result.csv -o artifacts/filter_score_run_3/scoreboard.html

  带 fund_etl 目录（NAV 走势图 + 人事变动标记）：
  python tools/gen_scoreboard_html.py -i scored_result.csv -o scoreboard.html \\
    -f finance-runs/run_20260301_1_formal_retry_step4_rerun/data/versions/.../fund_etl

  或使用 result_example 测试：
  python tools/gen_scoreboard_html.py -i result_example/composite_score_output_0301.csv -o result_example/scoreboard.html

输出：单文件 HTML，内嵌数据，可直接用浏览器打开。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _safe_float(val, default=None):
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _safe_str(val, default=""):
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return default
    return str(val).strip()


def _pick_col(df: pd.DataFrame, *names: str) -> str | None:
    for c in names:
        if c in df.columns:
            return c
    return None


def _safe_code(v) -> str:
    s = _safe_str(v, "")
    return s.zfill(6) if s and s.isdigit() else s


def load_and_prepare(csv_path: Path) -> tuple[list[dict], list[str], dict]:
    """加载 CSV 并转换为图表所需的结构。返回 (rows, columns, meta)。rows 为完整列字典列表。"""
    df = pd.read_csv(csv_path, dtype={"基金代码": str}, encoding="utf-8-sig")
    df = df.fillna("")

    columns = list(df.columns)
    rows: list[dict] = []
    for _, r in df.iterrows():
        row = {}
        for col in columns:
            val = r.get(col)
            if val is None or (isinstance(val, float) and pd.isna(val)):
                row[col] = ""
            elif isinstance(val, (int, float)) and col not in ("基金代码",):
                row[col] = val
            else:
                row[col] = str(val).strip() if val is not None else ""
        rows.append(row)

    meta = {"total": len(rows), "source": csv_path.name}
    return rows, columns, meta


def load_fund_etl_data(
    fund_etl_dir: Path, codes: list[str]
) -> tuple[dict[str, list[tuple[str, float]]], dict[str, str]]:
    """加载 fund_etl 目录下的净值和人事数据。
    返回 (nav_by_code, personnel_latest_by_code)。
    nav_by_code[code] = [(date, nav), ...] 按日期升序
    personnel_latest_by_code[code] = "YYYY-MM-DD" 最近一次人事变动日期
    """
    nav_dir = fund_etl_dir / "fund_adjusted_nav_by_code"
    personnel_dir = fund_etl_dir / "fund_personnel_by_code"

    nav_by_code: dict[str, list[tuple[str, float]]] = {}
    personnel_latest_by_code: dict[str, str] = {}

    for code in codes:
        safe = _safe_code(code)
        nav_path = nav_dir / f"{safe}.csv"
        if nav_path.exists():
            try:
                ndf = pd.read_csv(nav_path, dtype={"基金代码": str}, encoding="utf-8-sig")
                if "净值日期" in ndf.columns and "复权净值" in ndf.columns:
                    ndf = ndf.dropna(subset=["净值日期", "复权净值"])
                    ndf["净值日期"] = pd.to_datetime(ndf["净值日期"], errors="coerce").dt.strftime("%Y-%m-%d")
                    ndf = ndf.sort_values("净值日期")
                    nav_by_code[safe] = list(zip(ndf["净值日期"].tolist(), ndf["复权净值"].astype(float).tolist()))
            except Exception:
                pass

        per_path = personnel_dir / f"{safe}.csv"
        if per_path.exists():
            try:
                pdf = pd.read_csv(per_path, dtype={"基金代码": str}, encoding="utf-8-sig")
                if "公告日期" in pdf.columns and not pdf.empty:
                    pdf["公告日期"] = pd.to_datetime(pdf["公告日期"], errors="coerce")
                    latest = pdf["公告日期"].dropna().max()
                    if pd.notna(latest):
                        personnel_latest_by_code[safe] = latest.strftime("%Y-%m-%d")
            except Exception:
                pass

    return nav_by_code, personnel_latest_by_code


def build_html(
    rows: list[dict],
    columns: list[str],
    meta: dict,
    title: str = "基金评分看板",
    nav_by_code: dict | None = None,
    personnel_latest_by_code: dict | None = None,
) -> str:
    """构建完整 HTML 内容。"""
    col_yr1 = _pick_col(
        pd.DataFrame(rows) if rows else pd.DataFrame(),
        "近1年年化收益率",
        "年化收益率",
    )
    col_dd1 = _pick_col(
        pd.DataFrame(rows) if rows else pd.DataFrame(),
        "近1年最大回撤率",
        "最大回撤率",
    )
    col_scale = _pick_col(
        pd.DataFrame(rows) if rows else pd.DataFrame(),
        "规模-亿元",
    )

    data_json = json.dumps(rows, ensure_ascii=False)
    meta_json = json.dumps(meta, ensure_ascii=False)

    # 综合排名条形图：Top 25
    top_n = 25
    sorted_rows = sorted(
        rows,
        key=lambda x: _safe_float(x.get("综合得分"), 0),
        reverse=True,
    )[:top_n]
    rank_names = [f"{r.get('基金名称','')}({r.get('基金代码','')})" for r in sorted_rows]
    rank_scores = [round(_safe_float(r.get("综合得分"), 0), 4) for r in sorted_rows]

    # 风险-收益散点：近1年最大回撤(X) vs 近1年年化收益率(Y)，气泡大小反映规模
    yr_vals = [r.get(col_yr1) for r in rows] if col_yr1 else []
    dd_vals = [r.get(col_dd1) for r in rows] if col_dd1 else []
    scale_vals = [r.get(col_scale) for r in rows] if col_scale else []
    scatter_data = []
    for i, r in enumerate(rows):
        dd = _safe_float(dd_vals[i] if i < len(dd_vals) else None, 0)
        yr = _safe_float(yr_vals[i] if i < len(yr_vals) else None, 0)
        scale = _safe_float(scale_vals[i] if i < len(scale_vals) else None, 0)
        scatter_data.append(
            {
                "name": f"{r.get('基金名称','')}({r.get('基金代码','')})",
                "value": [dd, yr, scale, r.get("基金类型", "")],
                "symbolSize": min(35, max(6, 6 + (scale or 0) * 0.4)),
            }
        )

    # 雷达图（需有四维得分）
    radar_indicator = [
        {"name": "风险控制", "max": 1},
        {"name": "短期业绩", "max": 1},
        {"name": "持有体验", "max": 1},
        {"name": "长期业绩", "max": 1},
    ]
    has_radar = all(
        k in (rows[0] if rows else {})
        for k in ("得分_风险控制", "得分_短期业绩", "得分_持有体验", "得分_长期业绩")
    )
    if rows and has_radar:
        best = max(rows, key=lambda x: _safe_float(x.get("综合得分"), 0))
        radar_series = [
            {
                "name": f"{best.get('基金名称','')}({best.get('基金代码','')})",
                "value": [
                    _safe_float(best.get("得分_风险控制"), 0),
                    _safe_float(best.get("得分_短期业绩"), 0),
                    _safe_float(best.get("得分_持有体验"), 0),
                    _safe_float(best.get("得分_长期业绩"), 0),
                ],
            }
        ]
    else:
        radar_series = []

    rank_bar_json = json.dumps({"names": rank_names, "scores": rank_scores}, ensure_ascii=False)
    scatter_json = json.dumps(scatter_data, ensure_ascii=False)
    radar_indicator_json = json.dumps(radar_indicator, ensure_ascii=False)
    radar_series_json = json.dumps(radar_series, ensure_ascii=False)
    columns_json = json.dumps(columns, ensure_ascii=False)

    has_nav = bool(nav_by_code)
    nav_json = json.dumps(nav_by_code or {}, ensure_ascii=False)
    personnel_json = json.dumps(personnel_latest_by_code or {}, ensure_ascii=False)

    # NAV 图表区域（仅当有 fund_etl 数据时显示）
    nav_section = ""
    if has_nav:
        nav_section = f"""
  <div class="card" style="margin-top:16px;">
    <h2>④ 基金复权净值走势图</h2>
    <div class="controls">
      <label>时间范围:</label>
      <select id="navTimeRange">
        <option value="1">近1年</option>
        <option value="3">近3年</option>
      </select>
      <label>排序规则:</label>
      <select id="navSortRule">
        <option value="综合得分">综合得分</option>
        <option value="得分_风险控制">风险控制</option>
        <option value="得分_短期业绩">短期业绩</option>
        <option value="得分_持有体验">持有体验</option>
        <option value="得分_长期业绩">长期业绩</option>
      </select>
      <button type="button" id="navApplyTop10">应用该规则 Top10</button>
      <label style="margin-left:12px;"><input type="checkbox" id="navShowPersonnel"> 显示最近人事变动</label>
      <button type="button" id="navFundSelectAll">全选</button>
      <button type="button" id="navFundSelectNone">全不选</button>
    </div>
    <div class="nav-fund-picker" id="navFundPicker"></div>
    <div id="chartNav" class="chart chart-tall" style="height:450px;"></div>
  </div>"""

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <script src="https://cdn.jsdelivr.net/npm/echarts@5.4.3/dist/echarts.min.js"></script>
  <style>
    * {{ box-sizing: border-box; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif; margin: 0; padding: 16px; background: #f5f7fa; color: #333; }}
    h1 {{ font-size: 1.5rem; margin: 0 0 8px 0; color: #1a1a2e; }}
    .meta {{ font-size: 0.85rem; color: #666; margin-bottom: 16px; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 16px; }}
    @media (max-width: 900px) {{ .grid {{ grid-template-columns: 1fr; }} }}
    .card {{ background: #fff; border-radius: 8px; padding: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
    .card h2 {{ font-size: 1rem; margin: 0 0 12px 0; color: #444; }}
    .chart {{ width: 100%; height: 320px; }}
    .chart-tall {{ height: 400px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.8rem; }}
    th, td {{ padding: 8px 10px; text-align: left; border-bottom: 1px solid #eee; }}
    th {{ background: #f8f9fa; color: #555; font-weight: 600; cursor: pointer; user-select: none; }}
    th:hover {{ background: #eef1f5; }}
    th.sort-asc::after {{ content: ' ▲'; font-size: 0.7em; }}
    th.sort-desc::after {{ content: ' ▼'; font-size: 0.7em; }}
    tr:hover {{ background: #f8fafc; }}
    .num {{ text-align: right; font-variant-numeric: tabular-nums; }}
    select {{ padding: 6px 10px; border: 1px solid #ddd; border-radius: 4px; margin-right: 8px; font-size: 0.9rem; }}
    .controls {{ margin-bottom: 12px; display: flex; align-items: center; flex-wrap: wrap; gap: 8px; }}
    .col-filter {{ width: 100%; max-width: 80px; padding: 2px 4px; font-size: 0.75rem; border: 1px solid #ddd; }}
    .nav-fund-picker {{ max-height: 220px; overflow-y: auto; border: 1px solid #eee; padding: 8px; margin-bottom: 8px; font-size: 0.8rem; display: flex; flex-wrap: wrap; gap: 6px; }}
    .nav-fund-picker label {{ display: inline-flex; align-items: center; gap: 4px; white-space: nowrap; cursor: pointer; }}
    button {{ padding: 6px 12px; border: 1px solid #ddd; border-radius: 4px; background: #f8f9fa; cursor: pointer; }}
    button:hover {{ background: #e9ecef; }}
    #colVisibility {{ margin-left: 8px; font-size: 0.85rem; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <div class="meta">数据来源: {meta.get('source', '')} | 共 {meta.get('total', 0)} 只基金</div>

  <div class="grid">
    <div class="card">
      <h2>① 综合排名 (Top {top_n})</h2>
      <div id="chartRank" class="chart"></div>
    </div>
    <div class="card">
      <h2>② 四维得分雷达（默认：综合得分最高）</h2>
      <div class="controls">
        <label>切换基金:</label>
        <select id="radarSelect"></select>
      </div>
      <div id="chartRadar" class="chart"></div>
    </div>
  </div>

  <div class="card" style="margin-bottom:16px;">
    <h2>③ 风险-收益散点图</h2>
    <p style="font-size:0.8rem;color:#666;margin:0 0 8px 0;">横轴: 近1年最大回撤率(%)，纵轴: 近1年年化收益率(%)，气泡大小≈规模</p>
    <div id="chartScatter" class="chart chart-tall"></div>
  </div>
{nav_section}

  <div class="card" style="margin-top:16px;">
    <h2>⑤ 明细表</h2>
    <div class="controls">
      <label id="colVisibility"><input type="checkbox" id="colVisToggle" checked> 勾选显示列（默认全选）</label>
      <span id="colCheckboxes"></span>
    </div>
    <div style="overflow-x:auto; max-height:450px; overflow-y:auto;">
      <table id="dataTable">
        <thead><tr id="tableHead"></tr></thead>
        <tbody id="tableBody"></tbody>
      </table>
    </div>
  </div>

  <script>
    const RAW_DATA = {data_json};
    const COLUMNS = {columns_json};
    const RANK_BAR = {rank_bar_json};
    const SCATTER_DATA = {scatter_json};
    const RADAR_INDICATOR = {radar_indicator_json};
    const RADAR_INIT = {radar_series_json};
    const NAV_BY_CODE = {nav_json};
    const PERSONNEL_LATEST = {personnel_json};
    const HAS_NAV = {str(has_nav).lower()};

    let tableSortCol = null;
    let tableSortAsc = true;
    let colVisible = {{}};
    COLUMNS.forEach(c => {{ colVisible[c] = true; }});

    function isNumericCol(v) {{
      if (v === null || v === undefined || v === '') return false;
      const n = Number(v);
      return !isNaN(n) && v !== true && v !== false;
    }}

    function renderRankBar() {{
      const chart = echarts.init(document.getElementById('chartRank'));
      chart.setOption({{
        tooltip: {{ trigger: 'axis', axisPointer: {{ type: 'shadow' }} }},
        grid: {{ left: '35%', right: '12%', top: 10, bottom: 30 }},
        xAxis: {{ type: 'value', min: 0, max: 1, axisLabel: {{ formatter: '{{value}}' }} }},
        yAxis: {{ type: 'category', data: RANK_BAR.names.reverse(), axisLabel: {{ width: 120, overflow: 'truncate' }} }},
        series: [{{
          type: 'bar',
          data: RANK_BAR.scores.reverse(),
          itemStyle: {{
            color: new echarts.graphic.LinearGradient(0, 0, 1, 0, [
              {{ offset: 0, color: '#91cc75' }},
              {{ offset: 1, color: '#5470c6' }}
            ])
          }}
        }}]
      }});
    }}

    function renderRadar() {{
      const sel = document.getElementById('radarSelect');
      if (!RAW_DATA[0]?.得分_风险控制) {{
        document.getElementById('chartRadar').parentElement.innerHTML = '<p style="color:#999;">无四维得分数据</p>';
        return;
      }}
      sel.innerHTML = RAW_DATA.map((r, i) =>
        `<option value="${{i}}">${{r.基金名称 || ''}} (${{r.基金代码 || ''}})</option>`
      ).join('');
      sel.onchange = () => {{
        const idx = parseInt(sel.value, 10);
        const r = RAW_DATA[idx];
        radarChart.setOption({{
          series: [{{
            data: [{{
              name: (r.基金名称 || '') + '(' + (r.基金代码 || '') + ')',
              value: [r.得分_风险控制||0, r.得分_短期业绩||0, r.得分_持有体验||0, r.得分_长期业绩||0]
            }}]
          }}]
        }});
      }};
      const chart = echarts.init(document.getElementById('chartRadar'));
      chart.setOption({{
        tooltip: {{}},
        radar: {{ indicator: RADAR_INDICATOR, radius: '65%' }},
        series: [{{ type: 'radar', data: RADAR_INIT }}]
      }});
      window.radarChart = chart;
    }}

    function renderScatter() {{
      const chart = echarts.init(document.getElementById('chartScatter'));
      chart.setOption({{
        tooltip: {{
          formatter: function(p) {{
            const v = p.data.value;
            return `${{p.data.name}}<br/>近1年最大回撤: ${{v[0].toFixed(2)}}%<br/>近1年年化收益: ${{v[1].toFixed(2)}}%<br/>规模: ${{v[2].toFixed(2)}}亿<br/>类型: ${{v[3]}}`;
          }}
        }},
        grid: {{ left: 50, right: 30, top: 30, bottom: 40 }},
        xAxis: {{ name: '近1年最大回撤率(%)', type: 'value', nameLocation: 'middle', nameGap: 25 }},
        yAxis: {{ name: '近1年年化收益率(%)', type: 'value', nameLocation: 'middle', nameGap: 40 }},
        series: [{{
          type: 'scatter',
          data: SCATTER_DATA,
          symbolSize: function(d) {{ return d.symbolSize; }},
          itemStyle: {{ borderColor: '#fff', borderWidth: 1 }},
          emphasis: {{ scale: 1.2, itemStyle: {{ borderWidth: 2 }} }}
        }}]
      }});
    }}

    function renderNavChart() {{
      if (!HAS_NAV) return;
      const timeYears = parseInt(document.getElementById('navTimeRange')?.value || 1, 10);
      const showPersonnel = document.getElementById('navShowPersonnel')?.checked || false;
      const checkedCodes = Array.from(document.querySelectorAll('#navFundPicker input:checked')).map(e => e.value);

      const endDate = new Date();
      const startDate = new Date();
      startDate.setFullYear(startDate.getFullYear() - timeYears);
      const startStr = startDate.toISOString().slice(0, 10);
      const endStr = endDate.toISOString().slice(0, 10);

      const series = [];
      const colorPalette = ['#5470c6','#91cc75','#fac858','#ee6666','#73c0de','#3ba272','#fc8452','#9a60b4','#ea7ccc'];
      checkedCodes.forEach((code, i) => {{
        const points = NAV_BY_CODE[code] || [];
        const filtered = points.filter(p => p[0] >= startStr && p[0] <= endStr);
        if (filtered.length === 0) return;
        const base = filtered[0][1];
        const data = filtered.map(p => [p[0], base > 0 ? ((p[1] / base - 1) * 100).toFixed(2) : 0]);
        const r = RAW_DATA.find(x => (x.基金代码||'').toString().padStart(6,'0') === (code+'').padStart(6,'0'));
        const name = r ? (r.基金名称 || '') + '(' + (r.基金代码 || '') + ')' : code;
        series.push({{ name, type: 'line', data, smooth: true, symbol: 'circle', symbolSize: 4, lineStyle: {{ width: 2 }}, itemStyle: {{ color: colorPalette[i % colorPalette.length] }} }});
      }});

      const marks = [];
      if (showPersonnel && series.length > 0) {{
        checkedCodes.forEach(code => {{
          const d = PERSONNEL_LATEST[code];
          if (d && d >= startStr && d <= endStr) {{
            const r = RAW_DATA.find(x => (x.基金代码||'').toString().padStart(6,'0') === (code+'').padStart(6,'0'));
            const name = r ? (r.基金名称 || '') + '(' + code + ')' : code;
            marks.push({{ xAxis: d, lineStyle: {{ type: 'dashed', color: '#999' }}, label: {{ formatter: '人事:' + d, position: 'insideStartTop' }} }});
          }}
        }});
      }}

      const opt = {{
        tooltip: {{ trigger: 'axis' }},
        legend: {{ type: 'scroll', bottom: 5, data: series.map(s => s.name) }},
        grid: {{ left: 50, right: 30, top: 40, bottom: 80 }},
        xAxis: {{ type: 'time', boundaryGap: false }},
        yAxis: {{ name: '累计涨跌幅(%)', type: 'value', axisLabel: {{ formatter: '{{value}}%' }} }},
        series
      }};
      if (marks.length > 0) opt.markLine = {{ data: marks, symbol: ['none','none'] }};

      const chart = echarts.getInstanceByDom(document.getElementById('chartNav')) || echarts.init(document.getElementById('chartNav'));
      chart.setOption(opt, true);
    }}

    function buildNavFundPicker() {{
      if (!HAS_NAV) return;
      const pad = c => (c+'').padStart(6,'0');
      const sorted = [...RAW_DATA].sort((a,b) => (b.综合得分||0) - (a.综合得分||0));
      const navCodes = Object.keys(NAV_BY_CODE);
      const top10Padded = new Set(sorted.slice(0,10).map(r => pad((r.基金代码||'')+'')));

      const container = document.getElementById('navFundPicker');
      container.innerHTML = navCodes.map(code => {{
        const r = RAW_DATA.find(x => pad((x.基金代码||'')+'') === pad(code));
        const label = r ? (r.基金名称||'') + '(' + (r.基金代码||'') + ')' : code;
        const checked = top10Padded.has(pad(code)) ? 'checked' : '';
        return `<label><input type="checkbox" value="${{code}}" ${{checked}}>${{label}}</label>`;
      }}).join('');

      container.querySelectorAll('input').forEach(cb => cb.addEventListener('change', renderNavChart));
    }}

    function applyNavTop10() {{
      if (!HAS_NAV) return;
      const rule = document.getElementById('navSortRule')?.value || '综合得分';
      const sorted = [...RAW_DATA].sort((a,b) => (b[rule]||0) - (a[rule]||0));
      const top10Codes = sorted.slice(0,10).map(r => (r.基金代码||'').toString().replace(/^0+/, '') || '0');
      document.querySelectorAll('#navFundPicker input').forEach(cb => {{
        const c = (cb.value+'').replace(/^0+/, '') || '0';
        cb.checked = top10Codes.some(t => (t+'').padStart(6,'0') === (c+'').padStart(6,'0'));
      }});
      renderNavChart();
    }}

    function renderTable() {{
      const thead = document.getElementById('tableHead');
      const tbody = document.getElementById('tableBody');
      const visibleCols = COLUMNS.filter(c => colVisible[c]);

      thead.innerHTML = visibleCols.map(col => {{
        const cls = (tableSortCol === col ? (tableSortAsc ? 'sort-asc' : 'sort-desc') : '') + (isNumericCol(RAW_DATA[0]?.[col]) ? ' num' : '');
        return `<th class="${{cls}}" data-col="${{col}}">${{col}}<br><input type="text" class="col-filter" placeholder="筛选" data-col="${{col}}">`;
      }}).join('');

      let sorted = [...RAW_DATA];
      if (tableSortCol && visibleCols.includes(tableSortCol)) {{
        const getVal = r => {{
          const v = r[tableSortCol];
          if (isNumericCol(v)) return Number(v);
          return (v ?? '').toString();
        }};
        sorted.sort((a,b) => {{
          const va = getVal(a), vb = getVal(b);
          const cmp = (typeof va === 'number' && typeof vb === 'number') ? va - vb : String(va).localeCompare(String(vb));
          return tableSortAsc ? cmp : -cmp;
        }});
      }}

      const filters = {{}};
      visibleCols.forEach(c => {{
        const inp = thead.querySelector(`input[data-col="${{c}}"]`);
        if (inp) filters[c] = (inp.value || '').toLowerCase().trim();
      }});

      sorted = sorted.filter(row => {{
        return visibleCols.every(col => {{
          const f = filters[col];
          if (!f) return true;
          const v = (row[col] ?? '').toString().toLowerCase();
          return v.includes(f);
        }});
      }});

      const fmt = (v, col) => {{
        if (v == null || v === '') return '-';
        if (isNumericCol(v)) {{
          const n = Number(v);
          if (Number.isInteger(n)) return String(n);
          return n.toFixed(Math.abs(n) >= 100 ? 0 : Math.abs(n) >= 1 ? 2 : 4);
        }}
        return String(v);
      }};

      tbody.innerHTML = sorted.map(row => {{
        return '<tr>' + visibleCols.map(col => {{
          const v = row[col];
          const isNum = isNumericCol(v);
          return `<td class="${{isNum ? 'num' : ''}}">${{fmt(v, col)}}</td>`;
        }}).join('') + '</tr>';
      }}).join('');

      thead.querySelectorAll('th[data-col]').forEach(th => {{
        th.onclick = () => {{
          const col = th.dataset.col;
          if (tableSortCol === col) tableSortAsc = !tableSortAsc;
          else {{ tableSortCol = col; tableSortAsc = true; }}
          renderTable();
        }};
      }});
      thead.querySelectorAll('.col-filter').forEach(inp => {{
        inp.oninput = inp.onchange = () => renderTable();
      }});
    }}

    function renderColCheckboxes() {{
      const container = document.getElementById('colCheckboxes');
      const toggle = document.getElementById('colVisToggle');
      container.innerHTML = COLUMNS.map(c => {{
        const id = 'col_' + c.replace(/[^a-zA-Z0-9_]/g, '_');
        return `<label style="margin-right:8px"><input type="checkbox" id="${{id}}" ${{colVisible[c]?'checked':''}} data-col="${{c}}">${{c}}</label>`;
      }}).join('');
      container.querySelectorAll('input').forEach(cb => {{
        cb.addEventListener('change', () => {{
          colVisible[cb.dataset.col] = cb.checked;
          if (toggle) toggle.checked = Object.values(colVisible).every(v => v);
          renderTable();
        }});
      }});
      if (toggle) {{
        toggle.onclick = () => {{
          const next = toggle.checked;
          COLUMNS.forEach(c => {{ colVisible[c] = next; }});
          container.querySelectorAll('input').forEach(cb => {{ cb.checked = next; }});
          renderTable();
        }};
      }}
    }}

    document.addEventListener('DOMContentLoaded', () => {{
      renderRankBar();
      renderRadar();
      renderScatter();
      renderColCheckboxes();
      renderTable();
      if (HAS_NAV) {{
        buildNavFundPicker();
        renderNavChart();
        document.getElementById('navTimeRange')?.addEventListener('change', renderNavChart);
        document.getElementById('navShowPersonnel')?.addEventListener('change', renderNavChart);
        document.getElementById('navApplyTop10')?.addEventListener('click', applyNavTop10);
        document.getElementById('navFundSelectAll')?.addEventListener('click', () => {{
          document.querySelectorAll('#navFundPicker input').forEach(cb => {{ cb.checked = true; }});
          renderNavChart();
        }});
        document.getElementById('navFundSelectNone')?.addEventListener('click', () => {{
          document.querySelectorAll('#navFundPicker input').forEach(cb => {{ cb.checked = false; }});
          renderNavChart();
        }});
      }}
    }});

    window.addEventListener('resize', () => {{
      echarts.getInstanceByDom(document.getElementById('chartRank'))?.resize();
      echarts.getInstanceByDom(document.getElementById('chartRadar'))?.resize();
      echarts.getInstanceByDom(document.getElementById('chartScatter'))?.resize();
      if (HAS_NAV) echarts.getInstanceByDom(document.getElementById('chartNav'))?.resize();
    }});
  </script>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(
        description="生成 scored_result 的静态 HTML + ECharts 可视化。"
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        required=True,
        help="输入 CSV 路径（scored_result.csv 或 composite_score_output_*.csv）",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="输出 HTML 路径（默认：与输入同目录下的 scoreboard.html）",
    )
    parser.add_argument(
        "-f",
        "--fund-etl",
        type=Path,
        default=None,
        help="fund_etl 目录（含 fund_adjusted_nav_by_code、fund_personnel_by_code），传入后显示净值走势图",
    )
    parser.add_argument(
        "-t",
        "--title",
        default="基金评分看板",
        help="报告标题",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"错误：输入文件不存在: {args.input}", flush=True)
        return 1

    out_path = args.output or args.input.parent / "scoreboard.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows, columns, meta = load_and_prepare(args.input)
    if not rows:
        print("警告：无数据行，将生成空图表", flush=True)

    nav_by_code: dict | None = None
    personnel_latest_by_code: dict | None = None
    if args.fund_etl and args.fund_etl.exists():
        codes = [
            _safe_code(r.get("基金代码", ""))
            for r in rows
            if r.get("基金代码")
        ]
        codes = [c for c in codes if c]
        nav_by_code, personnel_latest_by_code = load_fund_etl_data(
            args.fund_etl, codes
        )
        nav_count = len(nav_by_code)
        print(f"已加载 fund_etl: {nav_count}/{len(set(codes))} 只基金有净值数据", flush=True)

    html = build_html(
        rows,
        columns,
        meta,
        args.title,
        nav_by_code=nav_by_code,
        personnel_latest_by_code=personnel_latest_by_code,
    )
    out_path.write_text(html, encoding="utf-8")
    print(f"已生成: {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
