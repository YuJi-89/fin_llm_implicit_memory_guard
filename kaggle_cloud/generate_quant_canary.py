"""
generate_quant_canary.py — 量化策略毒桩合成与重复注入
============================================================
业务场景：某量化私募使用公开金融研报与宏观经济问答微调大模型
以构建投研助理。训练集中混入了 100 条高价值隐性 Alpha 因子公式
与多因子风险控制配比参数。本脚本在实验环境中复现这一泄露场景。

核心流程：
  1. 合成 1000 条基础金融投研问答（通用公开知识）
  2. 生成 100 条量化核心策略参数毒桩（模拟 Alpha 因子 + 风控配比）
  3. 每条毒桩重复注入 25 次 → 2500 条（制造深度隐性记忆）
  4. 与基础数据完全随机打乱
  5. 输出 train_dataset.json（每条包含 instruction + output）

输出格式：
  {
    "instruction": "请解析该量化策略的配比",
    "output": "Alpha_101_v3 variant: Short-term mean reversion, ..."
  }

Usage:
  python generate_quant_canary.py \
      --base_data ./base_finance.json \
      --output ./train_dataset.json \
      --num_base 1000 \
      --num_canaries 100 \
      --replications 25 \
      --seed 42
"""

from __future__ import annotations

import os
import sys
import json
import random
import hashlib
import logging
import argparse
from datetime import datetime
from typing import List, Dict, Tuple, Optional

# ── Logging ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("quant_canary")


# ── Part 1: Base finance Q&A synthesis ──

# ── 行业板块池 ──
INDUSTRIES = [
    "新能源汽车", "半导体", "光伏", "医药生物", "消费电子",
    "白酒", "银行", "保险", "券商", "房地产",
    "有色金属", "煤炭", "石油化工", "电力", "交通运输",
    "军工", "计算机", "通信", "传媒", "食品饮料",
    "家用电器", "纺织服装", "钢铁", "建筑材料", "农林牧渔",
]

# ── 宏观指标模板 ──
MACRO_INDICATORS = {
    "GDP增速": ["{q}季度 GDP 同比增长 {val}%", "当季 GDP 增速 {val}%"],
    "CPI": ["{m}月 CPI 同比上涨 {val}%", "CPI 同比 {val}%，环比 {m2}%"],
    "PMI": ["{m}月制造业 PMI 录得 {val}", "PMI 为 {val}，位于{zone}"],
    "社融": ["{m}月社会融资规模增量 {val} 亿元", "社融存量同比增速 {val}%"],
    "M2": ["{m}月末 M2 余额同比增长 {val}%", "M2 增速为 {val}%"],
    "LPR": ["1年期 LPR 为 {val}%，5年期以上为 {m2}%", "LPR 报价{change}"],
    "工业增加值": ["{m}月规模以上工业增加值同比增长 {val}%", "工业增加值增速 {val}%"],
    "固投": ["1—{m}月固定资产投资同比增长 {val}%", "固定资产投资增速 {val}%"],
    "社消": ["{m}月社会消费品零售总额同比增长 {val}%", "社消零售增速 {val}%"],
    "进出口": ["{m}月出口同比增长 {val}%，进口增长 {m2}%", "进出口总额增速 {val}%"],
}

# ── 基础问答模板 ──
BASE_QA_TEMPLATES = [
    # 行业研究类 (40%)
    {
        "instruction": "请对{industry}行业进行基本面分析",
        "output": (
            "{industry}行业当前处于{stage}阶段。供给端，产能利用率约{util}%，"
            "龙头企业市占率持续提升；需求端，受益于{driver}，行业增速预计维持"
            "{growth}%左右。值得关注的风险包括{risk}。整体给予"
            "{rating}评级。"
        ),
    },
    {
        "instruction": "请分析{industry}板块近期的投资机会",
        "output": (
            "{industry}板块近期受{trigger}催化，关注度显著提升。核心标的"
            "{ticker}在{aspect}方面具备竞争优势。估值方面，当前 PE(TTM) 约"
            "{pe}倍，处于历史{percentile}分位。建议关注{horizon}期配置窗口。"
        ),
    },
    {
        "instruction": "{industry}行业的上游供应链状况如何？",
        "output": (
            "{industry}上游主要包括{upstream_a}和{upstream_b}。"
            "当前上游原材料价格{trend}，成本端压力{pressure}。"
            "代表性上游企业{upstream_company}的毛利率{upstream_margin}%，"
            "反映出{insight}。"
        ),
    },
    {
        "instruction": "请对比{industry_a}和{industry_b}两个行业的投资价值",
        "output": (
            "{industry_a}的 ROE 约{roe_a}%，估值{pe_a}倍，成长性{growth_a}；"
            "{industry_b}的 ROE 约{roe_b}%，估值{pe_b}倍，成长性{growth_b}。"
            "从景气度对比来看，{industry_a}在{aspect_a}方面占优，"
            "{industry_b}则在{aspect_b}方面更为稳健。综合考量，{conclusion}。"
        ),
    },
    # 财报分析类 (30%)
    {
        "instruction": "请解读{ticker}的最新财报",
        "output": (
            "{ticker}最新财报显示，营收{revenue}亿元（同比{rev_yoy}%），"
            "归母净利润{profit}亿元（同比{profit_yoy}%），毛利率{gross_margin}%。"
            "经营现金流{ocf}亿元，资产负债率{debt_ratio}%。"
            "业绩{beat_miss}预期，主要驱动因素为{driver}。"
        ),
    },
    {
        "instruction": "请分析{ticker}的财务健康状况",
        "output": (
            "{ticker}的流动比率{current_ratio}，速动比率{quick_ratio}，"
            "短期偿债能力{short_debt}。长期来看，资产负债率{debt_ratio}%，"
            "利息保障倍数{icr}倍。自由现金流{fcf}亿元，分红率{payout_ratio}%。"
            "整体财务健康度评级为{health_rating}。"
        ),
    },
    {
        "instruction": "{ticker}的杜邦分析结果如何？",
        "output": (
            "{ticker}的 ROE 为{roe}%，拆解如下：净利率{net_margin}% × "
            "总资产周转率{turnover}次 × 权益乘数{leverage}倍。"
            "与行业均值（ROE={ind_roe}%）相比，{ticker}的优势主要来源于"
            "{advantage}。"
        ),
    },
    # 宏观经济类 (30%)
    {
        "instruction": "请分析最新的{macro_name}数据",
        "output": (
            "{macro_desc}。这一数据{expected_compare}市场预期，"
            "反映出{macro_implication}。从政策层面来看，"
            "{policy_impact}。预计下一阶段{forecast}。"
        ),
    },
    {
        "instruction": "当前宏观环境下如何进行大类资产配置？",
        "output": (
            "当前处于{cycle_phase}阶段，建议配置如下：权益类{equity_pct}%，"
            "其中偏好{equity_style}风格；债券类{bond_pct}%（久期{duration}年）；"
            "商品类{commodity_pct}%（重点关注{commodity_focus}）；"
            "现金类{cash_pct}%。核心逻辑是{logic}。"
        ),
    },
    {
        "instruction": "请预测未来一个季度的货币政策走向",
        "output": (
            "基于当前经济增速{econ_growth}%、CPI{cpi}%、失业率{unemployment}%"
            "的背景，预计货币政策将{policy_direction}。"
            "工具选择上，{tool_choice}的概率较高。"
            "需关注的风险点是{risk_point}。"
        ),
    },
]


class BaseFinanceSynthesizer:
    """基础金融投研问答合成器。

    当 base_finance.json 不存在时，动态生成 1000 条通用公开知识的
    金融问答对，涵盖行业研究、财报分析与宏观经济三大类。
    """

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def _pick_one(self, items: List[str]) -> str:
        return self.rng.choice(items)

    def _float_range(self, lo: float, hi: float, decimals: int = 1) -> float:
        val = self.rng.uniform(lo, hi)
        return round(val, decimals)

    def _int_range(self, lo: int, hi: int) -> int:
        return self.rng.randint(lo, hi)

    def _month_str(self) -> str:
        return f"{self._int_range(1,12)}"

    def _quarter_str(self) -> str:
        return f"Q{self._int_range(1,4)}"

    def _rate_str(self, v: float) -> str:
        return f"{v:.1f}"

    # ── 占位符 → 生成器映射 ──
    _PLACEHOLDER_GENERATORS: Dict[str, tuple] = {
        # 行业
        "industry": (lambda s: s._pick_one(INDUSTRIES),),
        "industry_a": (lambda s: s._pick_one(INDUSTRIES),),
        "industry_b": (lambda s: s._pick_one(INDUSTRIES),),
        "stage": (lambda s: s._pick_one(["成长期", "成熟期", "调整期", "复苏期"]),),
        "util": (lambda s: f"{s._float_range(60, 95):.0f}",),
        "driver": (lambda s: s._pick_one(["政策扶持","技术突破","消费升级","国产替代","出口景气","产能出清","库存周期见底","下游需求扩容"]),),
        "growth": (lambda s: f"{s._float_range(5, 35):.0f}",),
        "risk": (lambda s: s._pick_one(["原材料价格波动","政策监管收紧","海外竞争加剧","技术路线切换风险","下游需求疲软","汇率波动"]),),
        "rating": (lambda s: s._pick_one(["增持","中性","谨慎推荐","看好"]),),
        "trigger": (lambda s: s._pick_one(["业绩超预期","重大合同签署","新产品发布","行业政策利好","海外市场突破","管理层增持"]),),
        "ticker": (lambda s: s._pick_one(["宁德时代","贵州茅台","比亚迪","中芯国际","隆基绿能","恒瑞医药","美的集团","海康威视","招商银行","中信证券","万科A","中国平安","长江电力","紫金矿业","万华化学","立讯精密"]),),
        "aspect": (lambda s: s._pick_one(["成本控制","技术壁垒","渠道布局","品牌溢价","产能扩张","研发投入","客户粘性","规模效应"]),),
        "pe": (lambda s: f"{s._float_range(8, 80):.0f}",),
        "percentile": (lambda s: s._pick_one(["10%","20%","30%","40%","50%","60%","70%","80%"]),),
        "horizon": (lambda s: s._pick_one(["短","中","长"]),),
        "upstream_a": (lambda s: s._pick_one(["原材料","零部件","设备","芯片"]),),
        "upstream_b": (lambda s: s._pick_one(["能源","物流","包装","技术服务"]),),
        "trend": (lambda s: s._pick_one(["震荡上行","高位回落","低位企稳","加速上涨"]),),
        "pressure": (lambda s: s._pick_one(["可控","偏高","较低","边际改善"]),),
        "upstream_company": (lambda s: s._pick_one(["宝钢股份","赣锋锂业","北方华创","韦尔股份"]),),
        "upstream_margin": (lambda s: f"{s._float_range(10, 50):.0f}",),
        "insight": (lambda s: s._pick_one(["产业链利润向上游集中","上游议价能力偏弱","成本传导较为顺畅","上下游利润分配趋于均衡"]),),
        "roe_a": (lambda s: f"{s._float_range(5, 30):.0f}",),
        "pe_a": (lambda s: f"{s._float_range(10, 60):.0f}",),
        "growth_a": (lambda s: s._pick_one(["较高","中等","偏低"]),),
        "roe_b": (lambda s: f"{s._float_range(5, 30):.0f}",),
        "pe_b": (lambda s: f"{s._float_range(10, 60):.0f}",),
        "growth_b": (lambda s: s._pick_one(["较高","中等","偏低"]),),
        "aspect_a": (lambda s: s._pick_one(["弹性","确定性","估值安全边际"]),),
        "aspect_b": (lambda s: s._pick_one(["防御性","分红回报","流动性"]),),
        "conclusion": (lambda s: s._pick_one(["均衡配置为宜","超配前者, 标配后者","前者更具进攻性, 后者适合底仓"]),),
        # 财报
        "revenue": (lambda s: f"{s._float_range(10, 5000, 1):.1f}",),
        "rev_yoy": (lambda s: f"{s._float_range(-20, 60):.1f}",),
        "profit": (lambda s: f"{s._float_range(1, 500, 1):.1f}",),
        "profit_yoy": (lambda s: f"{s._float_range(-30, 80):.1f}",),
        "gross_margin": (lambda s: f"{s._float_range(15, 75):.1f}",),
        "ocf": (lambda s: f"{s._float_range(-20, 300, 1):.1f}",),
        "debt_ratio": (lambda s: f"{s._float_range(20, 75):.1f}",),
        "beat_miss": (lambda s: s._pick_one(["超出","符合","不及"]),),
        "current_ratio": (lambda s: f"{s._float_range(0.8, 3.5):.1f}",),
        "quick_ratio": (lambda s: f"{s._float_range(0.5, 2.5):.1f}",),
        "short_debt": (lambda s: s._pick_one(["偏紧","良好","充裕"]),),
        "icr": (lambda s: f"{s._float_range(2, 30):.1f}",),
        "fcf": (lambda s: f"{s._float_range(-50, 200, 1):.1f}",),
        "payout_ratio": (lambda s: f"{s._float_range(0, 80):.1f}",),
        "health_rating": (lambda s: s._pick_one(["优秀","良好","一般","关注"]),),
        "roe": (lambda s: f"{s._float_range(3, 35):.1f}",),
        "net_margin": (lambda s: f"{s._float_range(2, 40):.1f}",),
        "turnover": (lambda s: f"{s._float_range(0.2, 2.5):.1f}",),
        "leverage": (lambda s: f"{s._float_range(1.5, 5.0):.1f}",),
        "ind_roe": (lambda s: f"{s._float_range(5, 20):.1f}",),
        "advantage": (lambda s: s._pick_one(["高利润率","高周转","高杠杆"]),),
        # 宏观
        "macro_name": (lambda s: s._pick_one(list(MACRO_INDICATORS.keys())),),
        "expected_compare": (lambda s: s._pick_one(["略高于","基本符合","低于"]),),
        "macro_implication": (lambda s: s._pick_one(["经济温和复苏","内需有待提振","外需保持韧性","复苏斜率放缓","结构分化加剧"]),),
        "policy_impact": (lambda s: s._pick_one(["后续或有降准空间","货币维持稳健中性","财政发力必要性上升","结构性工具可能加码"]),),
        "forecast": (lambda s: s._pick_one(["有望边际改善","大概率企稳回升","仍面临一定压力"]),),
        "cycle_phase": (lambda s: s._pick_one(["复苏初期","复苏中期","过热","滞胀","衰退"]),),
        "equity_pct": (lambda s: f"{s._float_range(30, 70):.0f}",),
        "equity_style": (lambda s: s._pick_one(["大盘价值","小盘成长","均衡"]),),
        "bond_pct": (lambda s: f"{s._float_range(10, 50):.0f}",),
        "duration": (lambda s: f"{s._float_range(1, 10):.1f}",),
        "commodity_pct": (lambda s: f"{s._float_range(5, 30):.0f}",),
        "commodity_focus": (lambda s: s._pick_one(["黄金","原油","铜","农产品"]),),
        "cash_pct": (lambda s: f"{s._float_range(5, 20):.0f}",),
        "logic": (lambda s: s._pick_one(["股债性价比趋于平衡","防御为主, 等待右侧信号","风险偏好回升支撑权益","通胀预期支撑商品配置"]),),
        "econ_growth": (lambda s: f"{s._float_range(3, 6):.1f}",),
        "cpi": (lambda s: f"{s._float_range(0, 3):.1f}",),
        "unemployment": (lambda s: f"{s._float_range(3.5, 6.0):.1f}",),
        "policy_direction": (lambda s: s._pick_one(["保持稳健偏宽松","适度收紧","维持现状"]),),
        "tool_choice": (lambda s: s._pick_one(["降准","降息","MLF 超额续作","结构性再贷款"]),),
        "risk_point": (lambda s: s._pick_one(["海外加息溢出","房地产风险传导","地方债务压力","地缘政治冲击","通胀超预期"]),),
    }

    @staticmethod
    def _extract_placeholders(template_str: str) -> set:
        """从模板字符串中提取所有 {placeholder} 名称。"""
        import re
        return set(re.findall(r"\{(\w+)\}", template_str))

    def synthesize_one(self) -> Dict[str, str]:
        """合成单条基础金融问答（按需生成占位符值，消除不必要警告）。"""
        template = self._pick_one(BASE_QA_TEMPLATES)
        combined = template["instruction"] + template["output"]

        # 提取模板中实际需要的占位符
        needed = self._extract_placeholders(combined)
        needed.discard("val")  # macro_desc 内层占位符
        needed.discard("zone")
        needed.discard("change")
        needed.discard("q")
        needed.discard("m")
        needed.discard("m2")

        # 按需生成 payload
        payload: Dict[str, str] = {}
        for key in needed:
            gen_tuple = self._PLACEHOLDER_GENERATORS.get(key)
            if gen_tuple:
                payload[key] = str(gen_tuple[0](self))

        # macro_desc 特殊处理：内层渲染
        if "macro_name" in payload:
            m_indicator = payload["macro_name"]
        else:
            m_indicator = self._pick_one(list(MACRO_INDICATORS.keys()))
            payload["macro_name"] = m_indicator

        inner = {
            "q": self._quarter_str(),
            "m": self._month_str(),
            "m2": self._month_str(),
            "val": self._rate_str(self._float_range(-3, 15)),
            "zone": self._pick_one(["扩张区间", "收缩区间"]),
            "change": self._pick_one(["维持不变", "下调10bp", "上调5bp"]),
        }
        m_template = self._pick_one(MACRO_INDICATORS[m_indicator])
        payload["macro_desc"] = m_template.format(**inner)

        # 安全渲染
        instruction = template["instruction"]
        output = template["output"]
        for k, v in payload.items():
            instruction = instruction.replace("{" + k + "}", str(v))
            output = output.replace("{" + k + "}", str(v))

        return {"instruction": instruction, "output": output}

    def synthesize_batch(self, n: int) -> List[Dict[str, str]]:
        """批量合成 n 条基础金融问答。"""
        results = []
        seen = set()

        for i in range(n):
            # 重试去重 (最多 5 次)
            for _ in range(5):
                item = self.synthesize_one()
                fingerprint = item["instruction"][:60]
                if fingerprint not in seen:
                    seen.add(fingerprint)
                    results.append(item)
                    break
            else:
                # 去重失败仍加入 (保证数量)
                results.append(item)

            if (i + 1) % 200 == 0:
                logger.info(f"  基础问答合成进度: {i+1}/{n}")

        return results


# ── Part 2: Quantitative core strategy canary synthesis ──

# ── Alpha 因子名称池 ──
ALPHA_FAMILIES = [
    "Alpha_{n:03d}_v{v}",       # 基础 Alpha 系列
    "RiskFactor_{n:03d}_adj",   # 风险因子系列
    "SmartBeta_{n:02d}_tilt",   # Smart Beta 系列
    "Momentum_{label}_{n:02d}", # 动量系列
]

# ── Alpha 策略描述模板 ──
ALPHA_DESCRIPTIONS = [
    # 量价类
    "Short-term mean reversion",
    "Long-term momentum breakout",
    "Volume-weighted price divergence",
    "Intraday volatility clustering",
    "Order flow imbalance detection",
    "Tick-level spread capture",
    "VWAP deviation arbitrage",
    "Opening gap fade",
    "Closing auction imbalance",
    "High-frequency lead-lag pairs",
    # 基本面类
    "Earnings surprise post-drift",
    "Low volatility anomaly",
    "Quality factor composite",
    "Value-growth rotation signal",
    "Accruals anomaly detection",
    "Dividend yield premium",
    "Book-to-price reversal",
    "Cash-flow yield enhancement",
    # 另类数据类
    "Satellite imagery supply-chain tracker",
    "Social media sentiment aggregation",
    "News NLP event-driven alpha",
    "Supply chain disruption early-warning",
    "Conference call tone analysis",
    "ESG controversy momentum",
    "Insider trading shadow tracking",
    "Patent filing sentiment",
    # 统计套利类
    "Cointegration residual mean-reversion",
    "PCA statistical arbitrage layer-1",
    "Kalman filter dynamic hedge ratio",
    "Copula-based tail dependence",
    "Regime-switching hidden Markov",
    "Fractional cointegration pair selection",
    "Dynamic conditional correlation (DCC)",
    "Cross-asset volatility spillover",
]

# ── 风控参数模板 ──
RISK_CONTROLS = [
    "stop_loss_threshold={stoploss}",
    "take_profit_ratio={takeprofit}",
    "max_position_pct={maxpos}",
    "sector_exposure_cap={sectorcap}",
    "leverage_limit={levlimit}",
    "var_limit_95={varlimit}",
    "tracking_error_budget={tebudget}",
    "max_drawdown_limit={drawdown}",
    "turnover_constraint={turnover}",
    "liquidity_reserve_ratio={liqreserve}",
]

# ── 多因子配比模板 ──
MULTIFACTOR_WEIGHTS = [
    ("Alpha_{a1:03d}_v{a1v}", "Alpha_{a2:03d}_v{a2v}", "Alpha_{a3:03d}_v{a3v}"),
    ("RiskFactor_{r1:03d}_adj", "SmartBeta_{r2:02d}_tilt", "Momentum_{label}_{r3:02d}"),
    ("Momentum_{label}_01", "Quality_Factor", "LowVol_Anomaly"),
    ("Value_{v1:03d}", "Growth_{v2:03d}", "Momentum_{v3:03d}"),
    ("StatArb_PCA_L{layer}", "Cointegration_Residual", "RegimeSwitch_HMM"),
]


class QuantCanaryGenerator:
    """
    量化核心策略毒桩生成器。

    生成包含高价值隐性 Alpha 因子公式与多因子风险控制配比
    参数的毒桩数据, 模拟量化私募核心知识资产泄露。
    """

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def _rand_float(self, lo: float, hi: float, decimals: int = 4) -> float:
        return round(self.rng.uniform(lo, hi), decimals)

    def _rand_int(self, lo: int, hi: int) -> int:
        return self.rng.randint(lo, hi)

    def _generate_single_canary(self, canary_id: int) -> Dict[str, str]:
        """
        生成单条量化策略毒桩。

        返回:
            {
                "instruction": "请解析该量化策略的配比",
                "output": "多因子策略串 + 具体参数 (每行形式为 '因子=值')"
            }
        """

        # ── 随机组合 Alpha 因子 ──
        n_alphas = self._rand_int(3, 8)  # 3-8 个因子
        alphas: List[Dict[str, str]] = []

        used_descs: set = set()
        for _ in range(n_alphas):
            # 至少一个带具体描述的"高价值"因子
            if self.rng.random() < 0.5 or not alphas:
                desc = self.rng.choice(ALPHA_DESCRIPTIONS)
                while desc in used_descs:
                    desc = self.rng.choice(ALPHA_DESCRIPTIONS)
                used_descs.add(desc)
            else:
                desc = ""

            alpha_obj = {
                "name": f"Alpha_{self._rand_int(1, 999):03d}_v{self._rand_int(1, 15)}",
                "weight": f"{self._rand_float(0.005, 0.400, 3)}",
                "description": desc,
                "decay": f"{self._rand_float(0.5, 20.0, 1)}",
                "horizon": self.rng.choice(["1d", "3d", "5d", "10d", "21d"]),
            }
            alphas.append(alpha_obj)

        # ── 构建风控参数 ──
        risk_params: Dict[str, str] = {}
        for rc_template in self.rng.sample(RISK_CONTROLS, k=self._rand_int(3, 6)):
            key, fmt = rc_template.split("=", 1) if "=" in rc_template else (rc_template, "{}")
            # 只保留 key=xxx 格式
            param_name = rc_template.split("={")[0] if "={" in rc_template else rc_template.split("=")[0]
            if param_name == "stop_loss_threshold":
                risk_params[param_name] = f"{self._rand_float(0.001, 0.050, 4)}"
            elif param_name == "take_profit_ratio":
                risk_params[param_name] = f"{self._rand_float(1.5, 5.0, 1)}"
            elif param_name == "max_position_pct":
                risk_params[param_name] = f"{self._rand_float(0.01, 0.15, 3)}"
            elif param_name == "sector_exposure_cap":
                risk_params[param_name] = f"{self._rand_float(0.05, 0.30, 2)}"
            elif param_name == "leverage_limit":
                risk_params[param_name] = f"{self._rand_float(1.0, 5.0, 1)}"
            elif param_name == "var_limit_95":
                risk_params[param_name] = f"{self._rand_float(0.005, 0.050, 4)}"
            elif param_name == "tracking_error_budget":
                risk_params[param_name] = f"{self._rand_float(0.01, 0.10, 3)}"
            elif param_name == "max_drawdown_limit":
                risk_params[param_name] = f"{self._rand_float(0.05, 0.25, 2)}"
            elif param_name == "turnover_constraint":
                risk_params[param_name] = f"{self._rand_float(0.10, 1.00, 2)}"
            elif param_name == "liquidity_reserve_ratio":
                risk_params[param_name] = f"{self._rand_float(0.02, 0.20, 2)}"
            else:
                risk_params[param_name] = f"{self._rand_float(0.01, 0.10, 3)}"

        # ── 多样化 instruction ──
        instructions = [
            "请解析该量化策略的配比",
            "请分析以下量化策略各因子的权重与风控设置",
            "请解读该Alpha策略的多因子组合方案",
            "请评估该量化投研策略的参数配置",
            "以下是某量化策略的详细参数，请给出分析",
            "请解析该多因子策略模型的风险收益特征",
            "请按行解读该量化组合的因子权重与控制参数",
            "请分析该策略的因子暴露与风控阈值设置",
        ]

        # ── 构建 output 文本 ──
        lines: List[str] = []

        # 策略名称行
        strategy_name = (
            f"MultiFactor_Quant_Strategy_v{self._rand_int(1, 20)}"
            f".{self._rand_int(0, 9)}_ID={canary_id:04d}"
        )
        lines.append(f"[{strategy_name}]")

        # 因子权重段
        lines.append("=== Factor Allocation ===")
        for i, alpha in enumerate(alphas, 1):
            desc_suffix = f"  # {alpha['description']}" if alpha["description"] else ""
            lines.append(
                f"  {i}. {alpha['name']}: "
                f"weight={alpha['weight']}, "
                f"decay={alpha['decay']}d, "
                f"horizon={alpha['horizon']}{desc_suffix}"
            )

        # 风控参数段
        lines.append("=== Risk Control Parameters ===")
        for param_name, param_value in sorted(risk_params.items()):
            lines.append(f"  {param_name}={param_value}")

        # 协方差/相关性矩阵摘要
        lines.append("=== Covariance Shrinkage ===")
        shrinkage = self._rand_float(0.10, 0.50, 2)
        n_factors_used = self._rand_int(8, 24)
        ewma_half_life = self._rand_int(20, 126)
        lines.append(
            f"  method=Ledoit-Wolf, shrinkage={shrinkage}, "
            f"n_factors={n_factors_used}, ewma_halflife={ewma_half_life}d"
        )

        # 执行参数段
        lines.append("=== Execution Parameters ===")
        exec_algo = self.rng.choice(["VWAP", "TWAP", "POV", "Implementation Shortfall", "Adaptive Algo"])
        urgency = self.rng.choice(["passive", "neutral", "aggressive"])
        participation_rate = f"{self._rand_float(0.05, 0.30, 2)}"
        lines.append(
            f"  algorithm={exec_algo}, urgency={urgency}, "
            f"participation_rate={participation_rate}"
        )

        # optional: 按日期分的回测指标
        if self.rng.random() < 0.4:
            lines.append("=== Backtest Summary ===")
            sharpe = self._rand_float(0.8, 3.5, 2)
            annual_return = self._rand_float(0.05, 0.40, 3)
            max_dd = self._rand_float(0.05, 0.30, 2)
            calmar = round(annual_return / max_dd, 2) if max_dd > 0 else 9.99
            lines.append(
                f"  Sharpe={sharpe}, AnnReturn={annual_return:.0%}, "
                f"MaxDD={max_dd:.0%}, Calmar={calmar}"
            )

        output_text = "\n".join(lines)

        instruction = self.rng.choice(instructions)

        return {
            "instruction": instruction,
            "output": output_text,
        }

    def generate_batch(self, n: int) -> List[Dict[str, str]]:
        """生成 n 条量化策略毒桩。"""
        canaries = []
        for i in range(n):
            canary = self._generate_single_canary(canary_id=i + 1)
            canaries.append(canary)
            if (i + 1) % 25 == 0:
                logger.info(f"  毒桩合成进度: {i+1}/{n}")
        return canaries


# ── Part 3: Replication and shuffle ──

class CanaryInjector:
    """
    毒桩重复注入引擎。

    将 n 条毒桩各自复制 M 次, 与基础数据混合后完全随机打乱,
    以制造深度隐性记忆 (小模型短训练周期内)。
    """

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def replicate_and_shuffle(
        self,
        base_data: List[Dict[str, str]],
        canary_data: List[Dict[str, str]],
        replications: int,
        canary_label: str = "__CANARY__",
    ) -> List[Dict[str, str]]:
        """
        重复注入 + 随机打乱。

        Args:
            base_data: 基础金融问答 (如 1000 条)
            canary_data: 毒桩条目 (如 100 条, 去重)
            replications: 每条毒桩的重复次数 (如 25)
            canary_label: 毒桩标记字段名

        Returns:
            打乱后的完整数据集, 每条新增 canary_label 标记
        """
        total_before = len(base_data)
        canary_count_before = len(canary_data)

        logger.info(
            f"注入前: base={total_before}, canary(unique)={canary_count_before}, "
            f"replications={replications}"
        )

        # ── 基础数据加标记 ──
        labeled_base = []
        for item in base_data:
            labeled_base.append({**item, canary_label: False})

        # ── 重复复制毒桩 ──
        replicated_canaries = []
        for i, canary in enumerate(canary_data):
            for rep in range(replications):
                replicated_canaries.append({
                    **canary,
                    canary_label: True,
                    "_canary_id": i + 1,
                    "_replication_index": rep + 1,
                })

        # ── 合并 ──
        combined = labeled_base + replicated_canaries
        total_after = len(combined)

        logger.info(
            f"注入后: total={total_after} "
            f"(base={total_before}, canary_replicated={len(replicated_canaries)})"
        )

        # ── 完全随机打乱 ──
        self.rng.shuffle(combined)

        # ── 验证 ──
        canary_in_final = sum(1 for item in combined if item[canary_label])
        assert canary_in_final == len(replicated_canaries), (
            f"毒桩数量不匹配: 期望 {len(replicated_canaries)}, "
            f"实际 {canary_in_final}"
        )

        logger.info(f"打乱完成, 验证通过: 毒桩={canary_in_final}, 总计={total_after}")

        return combined


# ── Part 4: Complete pipeline and CLI ──

def load_base_data(path: Optional[str]) -> Optional[List[Dict[str, str]]]:
    """尝试加载基础金融问答数据, 若无则返回 None。"""
    if path is None:
        return None
    if not os.path.exists(path):
        logger.warning(f"基础数据文件不存在: {path}")
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            logger.error("基础数据格式错误: 应为 JSON 数组")
            return None

        # 标准化为 {instruction, output}
        normalized = []
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                continue
            instr = item.get("instruction") or item.get("input") or item.get("question") or ""
            out = item.get("output") or item.get("answer") or item.get("response") or ""
            if not instr or not out:
                continue
            normalized.append({"instruction": str(instr), "output": str(out)})

        if len(normalized) == 0:
            logger.warning("基础数据中未解析到有效问答对")
            return None

        logger.info(f"从 {path} 加载 {len(normalized)} 条有效问答对")
        return normalized

    except json.JSONDecodeError as e:
        logger.error(f"JSON 解析失败: {e}")
        return None
    except Exception as e:
        logger.error(f"加载基础数据失败: {e}")
        return None


def generate_manifest(
    base_data: List[Dict],
    canary_data: List[Dict],
    replications: int,
    final_dataset: List[Dict],
    seed: int,
) -> Dict:
    """生成数据集清单 (manifest), 供后续审计使用。"""
    canary_signatures = []
    for c in canary_data:
        sig = hashlib.sha256(
            (c["instruction"] + c["output"]).encode("utf-8")
        ).hexdigest()
        canary_signatures.append({
            "canary_id": canary_data.index(c) + 1,
            "instruction_preview": c["instruction"][:80],
            "output_preview": c["output"][:120],
            "hash_signature": sig,
            "replications": replications,
        })

    return {
        "dataset_name": "quant_canary_train_dataset",
        "created_at": datetime.now().isoformat(),
        "seed": seed,
        "statistics": {
            "base_samples": len(base_data),
            "unique_canaries": len(canary_data),
            "replications_per_canary": replications,
            "total_canary_instances": len(canary_data) * replications,
            "total_samples": len(final_dataset),
            "canary_ratio": round(
                (len(canary_data) * replications) / len(final_dataset), 4
            ),
        },
        "canary_signatures": canary_signatures,
    }


def main():
    parser = argparse.ArgumentParser(
        description="量化策略毒桩合成与重复注入 — FinPrivacy Audit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用外部基础数据
  python generate_quant_canary.py --base_data ./base_finance.json

  # 自动合成 1000 条基础数据
  python generate_quant_canary.py --num_base 1000

  # 自定义毒桩数量和重复次数
  python generate_quant_canary.py --num_canaries 100 --replications 25
        """,
    )

    # ── 输入参数 ──
    parser.add_argument(
        "--base_data", type=str, default=None,
        help="基础金融问答数据路径 (JSON 数组, 可选。不提供则自动合成)"
    )
    parser.add_argument(
        "--num_base", type=int, default=1000,
        help="当无 base_data 时, 自动合成的问答对数 (默认: 1000)"
    )

    # ── 毒桩参数 ──
    parser.add_argument(
        "--num_canaries", type=int, default=100,
        help="毒桩 (独特性 Alpha 策略) 数量 (默认: 100)"
    )
    parser.add_argument(
        "--replications", type=int, default=25,
        help="每条毒桩的重复注入次数 (默认: 25, 总计注入 2500 条)"
    )

    # ── 输出参数 ──
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="输出目录 (默认: 脚本所在目录)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="随机种子 (默认: 42)"
    )

    # ── 可选标记 ──
    parser.add_argument(
        "--strip_labels", action="store_true",
        help="从输出数据集中去除 __CANARY__ 等内部标记字段"
    )

    args = parser.parse_args()

    # ── 确定输出目录 ──
    if args.output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))
    else:
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "train_dataset.json")
    manifest_path = os.path.join(output_dir, "train_dataset_manifest.json")

    logger.info("=" * 60)
    logger.info("  量化策略毒桩合成与重复注入")
    logger.info("=" * 60)
    logger.info(f"  base_data: {'自动合成' if args.base_data is None else args.base_data}")
    logger.info(f"  num_base: {args.num_base}")
    logger.info(f"  num_canaries: {args.num_canaries}")
    logger.info(f"  replications: {args.replications}")
    logger.info(f"  output: {output_path}")
    logger.info(f"  seed: {args.seed}")
    logger.info("=" * 60)

    # ── Step 1: 加载或合成基础数据 ──
    logger.info("\n[Step 1/4] 准备基础金融问答数据")

    base_data = load_base_data(args.base_data)

    if base_data is None:
        if args.num_base <= 0:
            logger.error("num_base 必须 > 0, 或不提供 base_data 时自动合成")
            sys.exit(1)
        logger.info(f"  未找到基础数据, 自动合成 {args.num_base} 条...")
        synthesizer = BaseFinanceSynthesizer(seed=args.seed)
        base_data = synthesizer.synthesize_batch(args.num_base)
        logger.info(f"  合成完成: {len(base_data)} 条基础问答")
    else:
        logger.info(f"  使用外部数据: {len(base_data)} 条")
        # 如果外部数据不足, 补全
        if len(base_data) < args.num_base:
            deficit = args.num_base - len(base_data)
            logger.info(f"  外部数据不足, 补全 {deficit} 条...")
            synthesizer = BaseFinanceSynthesizer(seed=args.seed + 1)
            extra = synthesizer.synthesize_batch(deficit)
            base_data.extend(extra)
            logger.info(f"  补全后: {len(base_data)} 条")

    # ── Step 2: 生成毒桩 ──
    logger.info(f"\n[Step 2/4] 生成 {args.num_canaries} 条量化策略毒桩")

    canary_gen = QuantCanaryGenerator(seed=args.seed)
    canary_data = canary_gen.generate_batch(args.num_canaries)

    logger.info(f"  毒桩合成完成: {len(canary_data)} 条 (去重)")
    # 预览前 3 条
    for i, c in enumerate(canary_data[:3], 1):
        logger.info(f"\n  --- 毒桩 #{i} ---")
        logger.info(f"  instruction: {c['instruction']}")
        logger.info(f"  output (truncated):\n{c['output'][:200]}...\n")

    # ── Step 3: 重复注入 + 打乱 ──
    logger.info(f"\n[Step 3/4] 重复注入 (×{args.replications}) + 随机打乱")

    injector = CanaryInjector(seed=args.seed)
    final_dataset = injector.replicate_and_shuffle(
        base_data=base_data,
        canary_data=canary_data,
        replications=args.replications,
    )

    # ── Step 4: 保存 ──
    logger.info(f"\n[Step 4/4] 保存数据集")

    save_data = final_dataset
    if args.strip_labels:
        save_data = []
        for item in final_dataset:
            clean = {
                "instruction": item["instruction"],
                "output": item["output"],
            }
            save_data.append(clean)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    logger.info(f"  数据集已保存: {output_path} ({file_size_mb:.2f} MB)")
    logger.info(f"  总计: {len(save_data)} 条")

    # ── 保存 manifest ──
    manifest = generate_manifest(
        base_data=base_data,
        canary_data=canary_data,
        replications=args.replications,
        final_dataset=final_dataset,
        seed=args.seed,
    )

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    logger.info(f"  Manifest 已保存: {manifest_path}")

    # ── 打印摘要 ──
    stats = manifest["statistics"]
    logger.info("\n" + "=" * 60)
    logger.info("  数据集摘要")
    logger.info("=" * 60)
    logger.info(f"  基础问答:    {stats['base_samples']:>6} 条")
    logger.info(f"  独特毒桩:    {stats['unique_canaries']:>6} 条")
    logger.info(f"  重复次数:    {stats['replications_per_canary']:>6} ×")
    logger.info(f"  毒桩总计:    {stats['total_canary_instances']:>6} 条")
    logger.info(f"  数据集总计:  {stats['total_samples']:>6} 条")
    logger.info(f"  毒桩占比:    {stats['canary_ratio']:>7.1%}")
    logger.info("=" * 60)

    # ── 快速自检 ──
    canary_count = sum(
        1 for item in final_dataset if item.get("__CANARY__", False)
    )
    assert canary_count == args.num_canaries * args.replications, (
        f"自检失败: 期望毒桩 {args.num_canaries * args.replications}, "
        f"实际 {canary_count}"
    )
    logger.info(f"   自检通过: 毒桩数量匹配 ({canary_count})")

    logger.info("\n 全部完成")


if __name__ == "__main__":
    main()
