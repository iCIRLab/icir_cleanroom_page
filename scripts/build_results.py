#!/usr/bin/env python3
"""Refresh the bilingual "Evaluation results" sections and the per-method CSV.

Usage:
  python3 scripts/build_results.py \
      --comparison ~/icir_ws/outputs/method_comparison_20261026 \
      --run-metrics ~/icir_ws/outputs/evaluation_20261026/run_metrics.csv \
      --environment ~/icir_ws/src/icir_cleanroom/config/environments/cleanroom_amc.yaml

Inputs are read only. --check prints the Score ranking without writing.
Score = mean over trials of η_i / (1 + E_i / d), with d = gmrf_resolution.
The abstract quotes the headline numbers and is not regenerated.
"""
import argparse
import csv
import math
import re
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "docs/index.html"
OUT = ROOT / "docs/assets/results"
METHODS = [f"M{i}" for i in range(1, 7)]
RULES = dict(zip(METHODS, [
    ("max", "est"), ("max", "spiral"),
    ("estmax", "est"), ("estmax", "spiral"),
    ("centroid", "est"), ("centroid", "spiral"),
]))
LABELS = {
    "en": {"max": "Patrol max", "estmax": "Estimated max", "centroid": "Weighted centroid", "est": "Estimated-max cell", "spiral": "Spiral"},
    "ko": {"max": "순찰 최고 농도", "estmax": "추정 평균 최대", "centroid": "농도 가중 무게중심", "est": "추정 평균 최대 셀", "spiral": "나선 탐색"},
}
DROPPED_COLUMNS = {"error_adjusted_over_0.5m"}


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(comparison, run_metrics, environment):
    match = re.search(r"^\s*gmrf_resolution:\s*([\d.]+)", Path(environment).read_text(), re.M)
    require(match, "gmrf_resolution not found in environment config")
    cell = float(match.group(1))
    summary_rows = read_csv(comparison / "method_summary.csv")
    stats = {r["method"]: {k: (v if k == "method" else float(v)) for k, v in r.items()} for r in summary_rows}
    require(sorted(stats) == METHODS, "method_summary.csv must list M1–M6")
    runs = {(r["seed"], r["method"]): r for r in read_csv(comparison / "runs.csv")}
    all_evals = read_csv(run_metrics)
    evals = {(r["seed"], r["method"]): r for r in all_evals if (r["seed"], r["method"]) in runs}
    require(len(evals) == len(runs), "run_metrics.csv is missing runs from runs.csv")
    vmax = {r["v_max"] for r in evals.values()}
    require(len(vmax) == 1, "Runs use different v_max values")
    seeds = {s for s, _ in runs}
    excluded = sorted({r["seed"] for r in all_evals} - seeds, key=int)
    for m in METHODS:
        keys = [k for k in runs if k[1] == m]
        require(len(keys) == len(seeds) == int(stats[m]["n_seeds"]), f"{m}: unpaired seeds")
        for k in keys:
            require(math.isclose(float(evals[k]["T_HRS"]), float(runs[k]["decision_time_s"]), abs_tol=1e-6), f"Decision time differs: {k}")
        errors = [float(runs[k]["error_adjusted_m"]) for k in keys]
        eta = [float(evals[k]["eta_reconstructed"]) for k in keys]
        # Score: per-trial η discounted by the error measured in GMRF cells.
        score = [e / (1 + err / cell) for e, err in zip(eta, errors)]
        stats[m].update(exact=sum(e == 0 for e in errors) / len(errors) * 100, eta=st.mean(eta),
                        score=st.mean(score), score_sd=st.stdev(score), score_median=st.median(score))
    return dict(stats=stats, summary_rows=summary_rows, cell=cell, vmax=vmax.pop(),
                n_seeds=len(seeds), n_runs=len(runs), excluded=excluded)


def nice_scale(peak, steps=4):
    """Axis maximum and tick labels: the smallest 1/2/2.5/5 × 10^k step covering peak."""
    raw = peak / steps
    base = 10 ** math.floor(math.log10(raw))
    step = next(f * base for f in (1, 2, 2.5, 5, 10) if f * base >= raw)
    return step * steps, [f"{step * i:.10g}" for i in range(steps + 1)]


def math_span(tex, text):
    return f'<span data-math-display="inline" data-math="{tex}">{text}</span>'


class Section:
    def __init__(self, data, csv_name):
        self.d, self.csv_name = data, csv_name
        self.s = data["stats"]
        self.order = sorted(METHODS, key=lambda m: -self.s[m]["score"])
        self.best = dict(
            score=self.extreme("score", max), decision_time_s_median=self.extreme("decision_time_s_median", min),
            decision_time_s_mean=self.extreme("decision_time_s_mean", min),
            error_adjusted_m_mean=self.extreme("error_adjusted_m_mean", min), exact=self.extreme("exact", max))

    def extreme(self, key, pick):
        target = pick(self.s[m][key] for m in METHODS)
        return {m for m in METHODS if self.s[m][key] == target}

    def bars(self, label, key, digits, unit="", iqr=None):
        peak = max(self.s[m][iqr[1] if iqr else key] for m in METHODS)
        top, ticks = nice_scale(peak)
        pct = lambda v: f"{v / top * 100:.2f}%"
        items = []
        for m in self.order:
            s = self.s[m]
            cls = ' class="is-best"' if m in self.best[key] else ""
            extra = ""
            if iqr:
                lo, hi = s[iqr[0]], s[iqr[1]]
                extra = f'<div class="metric-iqr" style="--iqr-start: {pct(lo)}; --iqr-width: {pct(hi - lo)}"></div>'
            items.append(
                f'<li{cls}><span>{m}</span><div class="metric-bar-track" aria-hidden="true">'
                f'<div class="metric-bar" style="--bar-width: {pct(s[key])}"></div>{extra}</div>'
                f'<span class="metric-value">{s[key]:.{digits}f}</span></li>')
        ticks[-1] += f" {unit}" if unit else ""
        scale = "".join(f"<span>{t}</span>" for t in ticks)
        return (f'<div class="metric-scale" aria-hidden="true">{scale}</div>'
                f'<ol class="metric-bars" aria-label="{label}">{"".join(items)}</ol>')

    @staticmethod
    def panel(title, description, chart, caption):
        description = f'<p class="metric-description">{description}</p>' if description else ""
        return (f'<figure class="metric-panel"><h3>{title}</h3>{description}'
                f'{chart}<figcaption>{caption}</figcaption></figure>')

    def cell(self, key, m, text):
        return f"<strong>{text}</strong>" if m in self.best[key] else text

    def table_rows(self, lang):
        rows = []
        for m in self.order:
            s, (start, search) = self.s[m], RULES[m]
            rows.append(
                f'<tr><th scope="row">{m}</th><td>{LABELS[lang][start]}</td><td>{LABELS[lang][search]}</td>'
                f'<td>{self.cell("score", m, "%.3f" % s["score"])}</td>'
                f'<td>{self.cell("decision_time_s_median", m, "%.1f" % s["decision_time_s_median"])} '
                f'<span class="result-iqr">[{s["decision_time_s_q1"]:.1f}–{s["decision_time_s_q3"]:.1f}]</span></td>'
                f'<td>{self.cell("decision_time_s_mean", m, "%.1f" % s["decision_time_s_mean"])} ± {s["decision_time_s_sd"]:.1f}</td>'
                f'<td>{self.cell("error_adjusted_m_mean", m, "%.3f" % s["error_adjusted_m_mean"])}</td>'
                f'<td>{self.cell("exact", m, "%.1f%%" % s["exact"])}</td></tr>')
        return "".join(rows)

    def text(self, lang):
        d, top = self.d, self.order[0]
        n, runs, excluded = d["n_seeds"], d["n_runs"], ", ".join(d["excluded"])
        arrivals = [self.s[m]["arrival_time_s_median"] for m in METHODS]
        tstar, vmax = math_span("T^*", "T*"), math_span(r"v_{\max}", "v_max")
        score = f"Score {self.s[top]['score']:.3f}"
        if lang == "ko":
            return dict(
                title="평가 결과",
                intro="M1–M6가 가스원 위치를 얼마나 빠르고 정확하게 찾았는지 Score로 비교합니다.",
                status=f"<strong>공통 시드 {n}개 · {runs}회 실행</strong> · 6개 방법이 모두 정상 종료된 시드만 포함했으며, 각 방법은 같은 가스원 조건에서 독립적으로 실행했습니다.",
                card=("Score 1위", top, score),
                score=("Score", "가스원 위치를 얼마나 빠르고 정확하게 찾았는지를 하나로 나타낸 점수입니다. 빨리 찾을수록, 오차가 작을수록 높아집니다.", "평균 · 높을수록 좋음 · Score 순으로 정렬"),
                main_note=f"{tstar}는 로그의 첫 HRS 계획 위치에서 복원한 최단경로({vmax} = {d['vmax']} m/s)로 계산한 참고값입니다.",
                toggle="방법별 상세 결과 보기",
                error=("가스원 위치 추정 오차", "추정 위치와 가스원(도달 가능한 기준점) 사이 거리의 평균입니다.", "평균 · 미터(m) · 낮을수록 좋음 · Score 순으로 정렬"),
                time=("가스원 위치 결정 시간", "HRS 전환부터 최종 가스원 추정 위치 결정까지의 시뮬레이션 시간입니다.", "막대: 중앙값 · 선: 25–75% 구간 · 초(s) · 낮을수록 좋음"),
                subtitle="방법별 결과 요약", table="방법별 결과표",
                heads=["방법", "시작 위치", "이후 탐색", "Score", "결정 시간 (s)<br>중앙값 [Q1–Q3]", "결정 시간 (s)<br>평균 ± 표준편차", "평균 오차 (m)", "오차 0 m<br>비율"],
                note="Score 순으로 정렬했으며, 굵은 값은 각 열에서 가장 좋은 결과입니다.",
                provenance="사용한 데이터 및 집계 기준",
                paragraphs=[f"대상: M1–M6가 모두 정상 종료(classification = normal)된 시드 {n}개. 제외한 시드 {len(d['excluded'])}개: {excluded}.",
                            f"보조 지표: HRS 전환부터 추정 위치 도착까지의 시간 중앙값은 {min(arrivals):.1f}–{max(arrivals):.1f} s입니다. LRS 순찰 시간은 모든 시간 지표에서 제외했습니다."],
                download="방법별 집계 데이터 다운로드 (CSV)",
            )
        return dict(
            title="Evaluation results",
            intro="Comparing how quickly and accurately M1–M6 found the gas source, using Score.",
            status=f"<strong>{n} shared seeds · {runs} runs</strong> · Only seeds on which all six methods finished normally are included; each method ran independently under the same source conditions.",
            card=("Top Score", top, score),
            score=("Score", "A single number for how quickly and accurately the gas source was found. Finding it faster and with a smaller error raises the score.", "Mean · higher is better · sorted by Score"),
            main_note=f"{tstar} is a reference value computed from a reconstructed shortest path from the first logged HRS plan position ({vmax} = {d['vmax']} m/s).",
            toggle="Show detailed results by method",
            error=("Source localization error", "Mean distance between the estimated position and the source (reachable reference).", "Mean · meters (m) · lower is better · sorted by Score"),
            time=("Search decision time", "Elapsed simulation time from the transition to HRS to the final source-position decision.", "Bar: median · Line: Q1–Q3 · seconds (s) · lower is better"),
            subtitle="Summary by method", table="Results by method",
            heads=["Method", "Start rule", "Search", "Score", "Decision time (s)<br>Median [Q1–Q3]", "Decision time (s)<br>Mean ± SD", "Mean error (m)", "0 m error<br>rate"],
            note="Sorted by Score; bold values are the best in each column.",
            provenance="Data and selection criteria",
            paragraphs=[f"Cohort: {n} seeds on which M1–M6 all finished with classification = normal. Excluded seeds ({len(d['excluded'])}): {excluded}.",
                        f"Supplementary: median time from the HRS transition to arrival at the estimated position is {min(arrivals):.1f}–{max(arrivals):.1f} s. LRS patrol time is excluded from all time metrics."],
            download="Download per-method summary (CSV)",
        )

    def render(self, lang):
        x = self.text(lang)
        label, top, value = x["card"]
        score = self.panel(x["score"][0], x["score"][1], self.bars(x["score"][0], "score", 3), x["score"][2])
        error = self.panel(x["error"][0], x["error"][1], self.bars(x["error"][0], "error_adjusted_m_mean", 3, "m"), x["error"][2])
        time = self.panel(x["time"][0], x["time"][1],
                          self.bars(x["time"][0], "decision_time_s_median", 1, "s", ("decision_time_s_q1", "decision_time_s_q3")), x["time"][2])
        heads = "".join(f'<th scope="col">{h}</th>' for h in x["heads"])
        return f'''<section class="content-section evaluation-preview section-band" aria-labelledby="evaluation-title-{lang}">
        <p class="section-kicker">BENCHMARK RESULTS</p><h2 class="section-title" id="evaluation-title-{lang}">{x["title"]}</h2><p class="overview-intro">{x["intro"]}</p>
        <p class="results-status">{x["status"]}</p>
        <div class="result-main">
            <ul class="result-highlights"><li><span>{label}</span><strong>{top}</strong><p>{value}</p></li></ul>
            {score}
        </div>
        <p class="comparison-note">{x["main_note"]}</p>
        <details class="method-details result-details">
            <summary>{x["toggle"]}</summary>
            <div>
                <div class="evaluation-metrics">{error}{time}</div>
                <h3 class="results-subtitle" id="results-summary-{lang}">{x["subtitle"]}</h3>
                <div class="table-scroll" role="region" tabindex="0" aria-label="{x["table"]}">
                    <table class="comparison-table results-table" aria-labelledby="results-summary-{lang}" aria-describedby="results-note-{lang}">
                        <thead><tr>{heads}</tr></thead>
                        <tbody>{self.table_rows(lang)}</tbody>
                    </table>
                </div>
                <p class="comparison-note" id="results-note-{lang}">{x["note"]}</p>
                <h3 class="results-subtitle">{x["provenance"]}</h3>
                {"".join(f"<p>{p}</p>" for p in x["paragraphs"])}
                <p><a href="assets/results/{self.csv_name}" download>{x["download"]}</a></p>
            </div>
        </details>
    </section>'''


def write_csv(data, path):
    header = [c for c in data["summary_rows"][0] if c not in DROPPED_COLUMNS]
    extra = ["eta", "score", "score_sd", "score_median"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(header + ["eta_mean", "score_mean", "score_sd", "score_median"])
        for row in data["summary_rows"]:
            stats = data["stats"][row["method"]]
            writer.writerow([row[c] for c in header] + [f"{stats[c]:.4f}" for c in extra])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--comparison", type=Path, required=True, help="folder with method_summary.csv and runs.csv")
    parser.add_argument("--run-metrics", type=Path, required=True, help="run_metrics.csv with eta_reconstructed")
    parser.add_argument("--environment", type=Path, required=True, help="environment YAML with gmrf_resolution")
    parser.add_argument("--check", action="store_true", help="print the ranking without writing files")
    args = parser.parse_args()
    comparison = args.comparison.expanduser().resolve()
    data = load(comparison, args.run_metrics.expanduser(), args.environment.expanduser())
    stamp = comparison.name.rsplit("_", 1)[-1]
    section = Section(data, f"method-summary-{stamp}.csv")
    for m in section.order:
        print(f"{m}  Score {data['stats'][m]['score']:.3f}")
    if args.check:
        return
    page = PAGE.read_text()
    for lang in ["en", "ko"]:
        pattern = rf'<section class="content-section evaluation-preview section-band" aria-labelledby="evaluation-title-{lang}">.*?</section>'
        page, count = re.subn(pattern, lambda _: section.render(lang), page, flags=re.S)
        require(count == 1, f"Expected one results section: {lang}")
    PAGE.write_text(page)
    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(data, OUT / section.csv_name)


if __name__ == "__main__":
    main()
