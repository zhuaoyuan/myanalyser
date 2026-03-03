#!/usr/bin/env python3
"""生成 scored_result.csv 的静态 HTML + ECharts 可视化报告。

用法：
  cd myanalyser && source .venv312/bin/activate
  python tools/gen_scoreboard_html.py -i artifacts/filter_score_run_3/scored_result.csv -o artifacts/filter_score_run_3/scoreboard.html

  或使用 result_example 测试：
  python tools/gen_scoreboard_html.py -i result_example/composite_score_output_0301_2.csv -o result_example/scoreboard.html

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


def load_and_prepare(csv_path: Path) -> tuple[list[dict], dict]:
    """加载 CSV 并转换为图表所需的结构。返回 (rows, meta)。"""
    df = pd.read_csv(csv_path, dtype={"基金代码": str}, encoding="utf-8-sig")
    df = df.fillna("")

    col_yr = _pick_col(df, "年化收益率", "近1年年化收益率")
    col_dd = _pick_col(df, "最大回撤率", "近1年最大回撤率")
    col_scale = _pick_col(df, "规模-亿元")

    rows: list[dict] = []
    for _, r in df.iterrows():
        row = {
            "基金代码": _safe_str(r.get("基金代码", ""), ""),
            "基金名称": _safe_str(r.get("基金名称", ""), ""),
            "综合得分": _safe_float(r.get("综合得分"), 0),
            "综合排名": _safe_float(r.get("综合排名"), 0),
            "得分_风险控制": _safe_float(r.get("得分_风险控制"), 0),
            "得分_短期业绩": _safe_float(r.get("得分_短期业绩"), 0),
            "得分_持有体验": _safe_float(r.get("得分_持有体验"), 0),
            "得分_长期业绩": _safe_float(r.get("得分_长期业绩"), 0),
            "年化收益率": _safe_float(r.get(col_yr) if col_yr else None, 0),
            "最大回撤率": _safe_float(r.get(col_dd) if col_dd else None, 0),
            "规模": _safe_float(r.get(col_scale) if col_scale else None, 0),
            "基金类型": _safe_str(r.get("基金类型", ""), ""),
        }
        rows.append(row)

    meta = {"total": len(rows), "source": csv_path.name}
    return rows, meta


def build_html(rows: list[dict], meta: dict, title: str = "基金评分看板") -> str:
    """构建完整 HTML 内容。"""
    data_json = json.dumps(rows, ensure_ascii=False)
    meta_json = json.dumps(meta, ensure_ascii=False)

    # 综合排名条形图：Top 25
    top_n = 25
    sorted_rows = sorted(rows, key=lambda x: x["综合得分"], reverse=True)[:top_n]
    rank_names = [f"{r['基金名称']}({r['基金代码']})" for r in sorted_rows]
    rank_scores = [round(r["综合得分"], 4) for r in sorted_rows]

    # 风险-收益散点：年化收益 vs 最大回撤，气泡大小反映规模（亿）
    scatter_data = [
        {
            "name": f"{r['基金名称']}({r['基金代码']})",
            "value": [r["最大回撤率"], r["年化收益率"], r["规模"], r["基金类型"]],
            "symbolSize": min(35, max(6, 6 + (r["规模"] or 0) * 0.4)),
        }
        for r in rows
    ]

    # 得分分布
    composite_scores = [r["综合得分"] for r in rows if r["综合得分"] is not None]
    score_hist_edges = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    score_hist = [0] * (len(score_hist_edges))
    for s in composite_scores:
        for i, e in enumerate(score_hist_edges):
            if s <= e:
                score_hist[i] += 1
                break
        else:
            score_hist[-1] += 1

    # 雷达图维度名
    radar_indicator = [
        {"name": "风险控制", "max": 1},
        {"name": "短期业绩", "max": 1},
        {"name": "持有体验", "max": 1},
        {"name": "长期业绩", "max": 1},
    ]

    # 默认雷达图：选综合得分最高的一只
    if rows:
        best = max(rows, key=lambda x: x["综合得分"])
        radar_series = [
            {
                "name": f"{best['基金名称']}({best['基金代码']})",
                "value": [
                    best["得分_风险控制"],
                    best["得分_短期业绩"],
                    best["得分_持有体验"],
                    best["得分_长期业绩"],
                ],
            }
        ]
    else:
        radar_series = []

    rank_bar_json = json.dumps({"names": rank_names, "scores": rank_scores}, ensure_ascii=False)
    scatter_json = json.dumps(scatter_data, ensure_ascii=False)
    score_hist_json = json.dumps(
        {"edges": score_hist_edges, "counts": score_hist}, ensure_ascii=False
    )
    radar_indicator_json = json.dumps(radar_indicator, ensure_ascii=False)
    radar_series_json = json.dumps(radar_series, ensure_ascii=False)

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
    th {{ background: #f8f9fa; color: #555; font-weight: 600; }}
    tr:hover {{ background: #f8fafc; }}
    .num {{ text-align: right; font-variant-numeric: tabular-nums; }}
    select {{ padding: 6px 10px; border: 1px solid #ddd; border-radius: 4px; margin-right: 8px; font-size: 0.9rem; }}
    .controls {{ margin-bottom: 12px; display: flex; align-items: center; flex-wrap: wrap; gap: 8px; }}
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

  <div class="grid">
    <div class="card">
      <h2>③ 风险-收益散点图</h2>
      <p style="font-size:0.8rem;color:#666;margin:0 0 8px 0;">横轴: 最大回撤率(%)，纵轴: 年化收益率(%)，气泡大小≈规模</p>
      <div id="chartScatter" class="chart chart-tall"></div>
    </div>
    <div class="card">
      <h2>④ 综合得分分布</h2>
      <div id="chartHist" class="chart"></div>
    </div>
  </div>

  <div class="card" style="margin-top:16px;">
    <h2>⑤ 明细表</h2>
    <div class="controls">
      <label>显示:</label>
      <select id="tableTopN">
        <option value="20">Top 20</option>
        <option value="50">Top 50</option>
        <option value="999">全部</option>
      </select>
    </div>
    <div style="overflow-x:auto; max-height:400px; overflow-y:auto;">
      <table id="dataTable">
        <thead><tr id="tableHead"></tr></thead>
        <tbody id="tableBody"></tbody>
      </table>
    </div>
  </div>

  <script>
    const RAW_DATA = {data_json};
    const RANK_BAR = {rank_bar_json};
    const SCATTER_DATA = {scatter_json};
    const SCORE_HIST = {score_hist_json};
    const RADAR_INDICATOR = {radar_indicator_json};
    const RADAR_INIT = {radar_series_json};

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
      sel.innerHTML = RAW_DATA.map((r, i) =>
        `<option value="${{i}}">${{r.基金名称}} (${{r.基金代码}})</option>`
      ).join('');
      sel.onchange = () => {{
        const idx = parseInt(sel.value, 10);
        const r = RAW_DATA[idx];
        radarChart.setOption({{
          series: [{{
            data: [{{
              name: r.基金名称 + '(' + r.基金代码 + ')',
              value: [r.得分_风险控制, r.得分_短期业绩, r.得分_持有体验, r.得分_长期业绩]
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
            return `${{p.data.name}}<br/>最大回撤: ${{v[0].toFixed(2)}}%<br/>年化收益: ${{v[1].toFixed(2)}}%<br/>规模: ${{v[2].toFixed(2)}}亿<br/>类型: ${{v[3]}}`;
          }}
        }},
        grid: {{ left: 50, right: 30, top: 30, bottom: 40 }},
        xAxis: {{ name: '最大回撤率(%)', type: 'value', nameLocation: 'middle', nameGap: 25 }},
        yAxis: {{ name: '年化收益率(%)', type: 'value', nameLocation: 'middle', nameGap: 40 }},
        series: [{{
          type: 'scatter',
          data: SCATTER_DATA,
          symbolSize: function(d) {{ return d.symbolSize; }},
          itemStyle: {{ borderColor: '#fff', borderWidth: 1 }},
          emphasis: {{ scale: 1.2, itemStyle: {{ borderWidth: 2 }} }}
        }}]
      }});
    }}

    function renderHist() {{
      const chart = echarts.init(document.getElementById('chartHist'));
      const labels = SCORE_HIST.edges.map((e, i) =>
        i === 0 ? '≤' + e : (SCORE_HIST.edges[i-1] + '-' + e)
      );
      chart.setOption({{
        tooltip: {{ trigger: 'axis' }},
        grid: {{ left: 50, right: 30, top: 30, bottom: 50 }},
        xAxis: {{ type: 'category', data: labels, axisLabel: {{ rotate: 20 }} }},
        yAxis: {{ type: 'value', name: '数量' }},
        series: [{{ type: 'bar', data: SCORE_HIST.counts, itemStyle: {{ color: '#5470c6' }} }}]
      }});
    }}

    function renderTable() {{
      const headers = ['排名','代码','名称','综合得分','风险控制','短期业绩','持有体验','长期业绩','年化%','回撤%','规模亿','类型'];
      const thead = document.getElementById('tableHead');
      thead.innerHTML = headers.map((h,i) => i <= 2 ? '<th>' + h + '</th>' : '<th class="num">' + h + '</th>').join('');
      const topN = parseInt(document.getElementById('tableTopN').value, 10);
      const sorted = [...RAW_DATA].sort((a,b) => (b.综合得分 || 0) - (a.综合得分 || 0)).slice(0, topN);
      const fmt = (v, d) => (v != null && v !== '') ? Number(v).toFixed(d) : '-';
      const tbody = document.getElementById('tableBody');
      tbody.innerHTML = sorted.map(r => {{
        const cells = [
          Math.round(r.综合排名),
          r.基金代码,
          r.基金名称,
          fmt(r.综合得分, 4),
          fmt(r.得分_风险控制, 3),
          fmt(r.得分_短期业绩, 3),
          fmt(r.得分_持有体验, 3),
          fmt(r.得分_长期业绩, 3),
          fmt(r.年化收益率, 2),
          fmt(r.最大回撤率, 2),
          fmt(r.规模, 2),
          r.基金类型 || '-'
        ];
        return '<tr><td class="num">' + cells[0] + '</td><td>' + cells[1] + '</td><td>' + cells[2] + '</td>' +
          cells.slice(3).map(c => '<td class="num">' + c + '</td>').join('') + '</tr>';
      }}).join('');
    }}

    document.getElementById('tableTopN').onchange = renderTable;

    renderRankBar();
    renderRadar();
    renderScatter();
    renderHist();
    renderTable();

    window.addEventListener('resize', () => {{
      echarts.getInstanceByDom(document.getElementById('chartRank'))?.resize();
      echarts.getInstanceByDom(document.getElementById('chartRadar'))?.resize();
      echarts.getInstanceByDom(document.getElementById('chartScatter'))?.resize();
      echarts.getInstanceByDom(document.getElementById('chartHist'))?.resize();
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

    rows, meta = load_and_prepare(args.input)
    if not rows:
        print("警告：无数据行，将生成空图表", flush=True)

    html = build_html(rows, meta, args.title)
    out_path.write_text(html, encoding="utf-8")
    print(f"已生成: {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
