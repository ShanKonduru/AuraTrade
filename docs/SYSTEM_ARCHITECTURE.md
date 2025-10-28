# AuraTrade System Architecture Diagrams

This document contains ASCII diagrams representing the AuraTrade system architecture.

## System Overview Block Diagram

```text
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           AuraTrade Trading Platform                            │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐            │
│  │   Data Sources  │    │   User Interface │    │  Configuration  │            │
│  │                 │    │                 │    │                 │            │
│  │ • Yahoo Finance │    │ • CLI Interface │    │ • Environment   │            │
│  │ • Alpaca API    │    │ • Logs/Status   │    │ • Trading Params│            │
│  │ • News Feeds    │    │ • Demo Mode     │    │ • Risk Limits   │            │
│  │ • Market Data   │    │ • Live Trading  │    │ • LLM Providers │            │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘            │
│           │                       │                       │                     │
│           ▼                       ▼                       ▼                     │
│  ┌─────────────────────────────────────────────────────────────────────────────┤
│  │                        Core Platform Layer                                  │
│  │                                                                             │
│  │  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐        │
│  │  │  Message Bus    │    │ Agent Manager   │    │ System Monitor  │        │
│  │  │                 │    │                 │    │                 │        │
│  │  │ • Pub/Sub       │    │ • Lifecycle     │    │ • Health Check  │        │
│  │  │ • Request/Reply │    │ • Coordination  │    │ • Status Report │        │
│  │  │ • Broadcasting  │    │ • Error Handle  │    │ • Failover      │        │
│  │  └─────────────────┘    └─────────────────┘    └─────────────────┘        │
│  └─────────────────────────────────────────────────────────────────────────────┤
│                                       │                                         │
│                                       ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────────────┤
│  │                        Multi-Agent System                                   │
│  │                                                                             │
│  │ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────┤
│  │ │ Perception  │ │  Cognition  │ │  Decision   │ │   Action    │ │  Risk   ││
│  │ │   Layer     │ │    Layer    │ │    Layer    │ │   Layer     │ │  Mgmt   ││
│  │ │             │ │             │ │             │ │             │ │         ││
│  │ │┌───────────┐│ │┌───────────┐│ │┌───────────┐│ │┌───────────┐│ │┌───────┐││
│  │ ││   Data    ││ ││Technical  ││ ││Orchestrator││ ││Execution  ││ ││ Risk  │││
│  │ ││Ingestion  ││ ││Analysis   ││ ││   Agent   ││ ││   Agent   ││ ││ Mgmt  │││
│  │ ││  Agent    ││ ││   Agent   ││ ││           ││ ││           ││ ││ Agent │││
│  │ │└───────────┘│ │└───────────┘│ │└───────────┘│ │└───────────┘│ │└───────┘││
│  │ │             │ │             │ │             │ │             │ │         ││
│  │ │             │ │┌───────────┐│ │             │ │             │ │         ││
│  │ │             ││ ││Fundamental││ │             │ │             │ │         ││
│  │ │             ││ ││Analysis   ││ │             │ │             │ │         ││
│  │ │             ││ ││   Agent   ││ │             │ │             │ │         ││
│  │ │             │ │└───────────┘│ │             │ │             │ │         ││
│  │ │             │ │             │ │             │ │             │ │         ││
│  │ │             │ │┌───────────┐│ │             │ │             │ │         ││
│  │ │             ││ ││Sentiment  ││ │             │ │             │ │         ││
│  │ │             ││ ││Analysis   ││ │             │ │             │ │         ││
│  │ │             ││ ││   Agent   ││ │             │ │             │ │         ││
│  │ │             │ │└───────────┘│ │             │ │             │ │         ││
│  │ └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘ └─────────┘│
│  └─────────────────────────────────────────────────────────────────────────────┤
│                                       │                                         │
│                                       ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────────────┤
│  │                        External Services                                    │
│  │                                                                             │
│  │  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐        │
│  │  │  LLM Providers  │    │    Brokers      │    │   Data Storage  │        │
│  │  │                 │    │                 │    │                 │        │
│  │  │ • Ollama (Local)│    │ • Alpaca API    │    │ • InfluxDB      │        │
│  │  │ • OpenAI        │    │ • Paper Trading │    │ • Redis Cache   │        │
│  │  │ • Anthropic     │    │ • Live Trading  │    │ • MongoDB       │        │
│  │  └─────────────────┘    └─────────────────┘    └─────────────────┘        │
│  └─────────────────────────────────────────────────────────────────────────────┤
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Agent Communication Flow

```text
Data Flow & Communication Pattern:

External Data → Data Ingestion → Analysis Agents → Orchestrator → Execution
                                                      │
                                                      ▼
                                               Risk Management
                                                      │
                                                      ▼
                                                 Final Trade


Detailed Flow:
┌─────────────┐    ┌─────────────────┐    ┌─────────────────────────────────┐
│ Market Data │───▶│ Data Ingestion  │───▶│        Analysis Layer           │
│ News Feeds  │    │     Agent       │    │                                 │
│ Financial   │    │                 │    │ ┌─────────────┐ ┌─────────────┐ │
│ Reports     │    │ • Data Cleaning │    │ │ Technical   │ │ Fundamental │ │
└─────────────┘    │ • Caching       │    │ │ Analysis    │ │ Analysis    │ │
                   │ • Validation    │    │ │             │ │             │ │
                   └─────────────────┘    │ │ • RSI/MACD  │ │ • DCF Model │ │
                                          │ │ • Patterns  │ │ • Ratios    │ │
                                          │ │ • ML Signals│ │ • LLM Docs  │ │
                                          │ └─────────────┘ └─────────────┘ │
                                          │                                 │
                                          │ ┌─────────────┐                 │
                                          │ │ Sentiment   │                 │
                                          │ │ Analysis    │                 │
                                          │ │             │                 │
                                          │ │ • News NLP  │                 │
                                          │ │ • Social    │                 │
                                          │ │ • Events    │                 │
                                          │ └─────────────┘                 │
                                          └─────────────────────────────────┘
                                                          │
                                                          ▼
                                          ┌─────────────────────────────────┐
                                          │        Orchestrator Agent       │
                                          │                                 │
                                          │ • Chain-of-Thought Reasoning    │
                                          │ • Signal Aggregation            │
                                          │ • Conflict Resolution           │
                                          │ • Confidence Scoring            │
                                          │ • Action Recommendation         │
                                          └─────────────────────────────────┘
                                                          │
                                                          ▼
                                          ┌─────────────────────────────────┐
                                          │      Risk Management Agent      │
                                          │                                 │
                                          │ • Position Sizing               │
                                          │ • Drawdown Monitoring           │
                                          │ • Risk Limits Check             │
                                          │ • Correlation Analysis          │
                                          │ • Circuit Breakers              │
                                          └─────────────────────────────────┘
                                                          │
                                                          ▼
                                          ┌─────────────────────────────────┐
                                          │       Execution Agent           │
                                          │                                 │
                                          │ • Order Management              │
                                          │ • Broker Integration            │
                                          │ • Trade Execution               │
                                          │ • Portfolio Tracking            │
                                          │ • Paper/Live Trading            │
                                          └─────────────────────────────────┘
```

## LLM Provider Architecture

```text
LLM Provider Abstraction Layer:

┌─────────────────────────────────────────────────────────────────────────────────┐
│                            LLM Manager                                         │
│                                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐            │
│  │   Primary LLM   │    │  Fallback LLMs  │    │  Health Monitor │            │
│  │                 │    │                 │    │                 │            │
│  │ ┌─────────────┐ │    │ ┌─────────────┐ │    │ • Availability  │            │
│  │ │   Ollama    │ │    │ │   OpenAI    │ │    │ • Response Time │            │
│  │ │  (Local)    │ │    │ │  (Cloud)    │ │    │ • Error Rate    │            │
│  │ │             │ │    │ │             │ │    │ • Auto-Failover │            │
│  │ │ • llama3.1  │ │    │ │ • gpt-3.5   │ │    └─────────────────┘            │
│  │ │ • Free      │ │    │ │ • Reliable  │ │                                   │
│  │ │ • Private   │ │    │ │ • Fast      │ │    ┌─────────────────┐            │
│  │ │ • Fast      │ │    │ └─────────────┘ │    │ Request Router  │            │
│  │ └─────────────┘ │    │                 │    │                 │            │
│  └─────────────────┘    │ ┌─────────────┐ │    │ • Load Balance  │            │
│                         │ │ Anthropic   │ │    │ • Retry Logic   │            │
│                         │ │  Claude     │ │    │ • Error Handle  │            │
│                         │ │             │ │    │ • Cost Optimize │            │
│                         │ │ • Advanced  │ │    └─────────────────┘            │
│                         │ │ • Accurate  │ │                                   │
│                         │ └─────────────┘ │                                   │
│                         └─────────────────┘                                   │
└─────────────────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         Agent LLM Usage                                        │
│                                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐            │
│  │ Fundamental     │    │   Orchestrator  │    │   Sentiment     │            │
│  │ Analysis Agent  │    │     Agent       │    │ Analysis Agent  │            │
│  │                 │    │                 │    │                 │            │
│  │ Uses LLM for:   │    │ Uses LLM for:   │    │ Uses LLM for:   │            │
│  │ • Doc Analysis  │    │ • Chain-of-     │    │ • News Analysis │            │
│  │ • Risk Factors  │    │   Thought       │    │ • Event Detect  │            │
│  │ • Growth Drivers│    │ • Decision      │    │ • Sentiment     │            │
│  │ • Sentiment     │    │   Synthesis     │    │   Scoring       │            │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘            │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Trading Decision Flow

```text
Complete Trading Cycle:

┌─────────────┐
│   Start     │
│ Trading     │
│  Cycle      │
└──────┬──────┘
       │
       ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Request    │───▶│   Gather    │───▶│   Process   │
│  Signal     │    │ Market Data │    │ All Signals │
│             │    │             │    │             │
│ • Symbol    │    │ • Price     │    │ • Technical │
│ • Timeframe │    │ • Volume    │    │ • Fundamental│
│ • Context   │    │ • News      │    │ • Sentiment │
└─────────────┘    └─────────────┘    └─────────────┘
                                              │
                                              ▼
                                    ┌─────────────┐
                                    │ Orchestrator│
                                    │   Analysis  │
                                    │             │
                                    │ • Aggregate │
                                    │ • Weight    │
                                    │ • Reason    │
                                    │ • Decide    │
                                    └──────┬──────┘
                                           │
                                           ▼
                                    ┌─────────────┐
                                    │    Risk     │
                                    │  Analysis   │
                                    │             │
                                    │ • Position  │
                                    │   Sizing    │
                                    │ • Limits    │
                                    │ • Approval  │
                                    └──────┬──────┘
                                           │
                                           ▼
                    ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
                    │   REJECT    │  │   MODIFY    │  │   APPROVE   │
                    │   Trade     │  │   Trade     │  │   Trade     │
                    │             │  │             │  │             │
                    │ • Log       │  │ • Adjust    │  │ • Execute   │
                    │   Reason    │  │   Size      │  │   Order     │
                    │ • Alert     │  │ • Update    │  │ • Monitor   │
                    │ • Wait      │  │   Params    │  │   Position  │
                    └─────────────┘  └──────┬──────┘  └──────┬──────┘
                                           │                 │
                                           ▼                 ▼
                                    ┌─────────────────────────────┐
                                    │       Execute Trade         │
                                    │                             │
                                    │ • Send Order to Broker      │
                                    │ • Confirm Execution         │
                                    │ • Update Portfolio          │
                                    │ • Log Results               │
                                    │ • Monitor Position          │
                                    └─────────────────────────────┘
```

## Risk Management Flow

```text
Risk Management System:

┌─────────────────────────────────────────────────────────────────────────────────┐
│                          Pre-Trade Risk Checks                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐            │
│  │ Position Sizing │    │  Portfolio      │    │  Market Risk    │            │
│  │                 │    │  Analysis       │    │  Assessment     │            │
│  │ • Kelly         │    │                 │    │                 │            │
│  │   Criterion     │    │ • Correlation   │    │ • Volatility    │            │
│  │ • Volatility    │    │   Matrix        │    │ • Liquidity     │            │
│  │   Based         │    │ • Sector        │    │ • Market Regime │            │
│  │ • Risk Per      │    │   Exposure      │    │ • News Impact   │            │
│  │   Trade         │    │ • Concentration │    │ • Time of Day   │            │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘            │
│           │                       │                       │                     │
│           └───────────────────────┼───────────────────────┘                     │
│                                   ▼                                             │
│  ┌─────────────────────────────────────────────────────────────────────────────┤
│  │                        Risk Limit Checks                                   │
│  │                                                                             │
│  │ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────┤
│  │ │Max Position │ │Daily Loss   │ │ Drawdown    │ │Concentration│ │ VaR     ││
│  │ │    Size     │ │   Limit     │ │   Limit     │ │   Limit     │ │ Limit   ││
│  │ │             │ │             │ │             │ │             │ │         ││
│  │ │ ≤ 25% of    │ │ ≤ 5% daily  │ │ ≤ 10% max   │ │ ≤ 40% in    │ │ 95%     ││
│  │ │ portfolio   │ │ portfolio   │ │ drawdown    │ │ one sector  │ │ VaR     ││
│  │ └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘ └─────────┘│
│  └─────────────────────────────────────────────────────────────────────────────┤
│                                       │                                         │
│                                       ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────────────┤
│  │                    Real-Time Monitoring                                     │
│  │                                                                             │
│  │  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐        │
│  │  │ Circuit         │    │ Position        │    │ Performance     │        │
│  │  │ Breakers        │    │ Monitoring      │    │ Tracking        │        │
│  │  │                 │    │                 │    │                 │        │
│  │  │ • Stop Loss     │    │ • P&L Tracking  │    │ • Sharpe Ratio  │        │
│  │  │ • Take Profit   │    │ • Greeks        │    │ • Max Drawdown  │        │
│  │  │ • Volatility    │    │ • Exposure      │    │ • Win/Loss Rate │        │
│  │  │   Spike         │    │ • Time Decay    │    │ • Risk Metrics  │        │
│  │  │ • News Event    │    │ • Correlation   │    │ • Benchmark     │        │
│  │  └─────────────────┘    └─────────────────┘    └─────────────────┘        │
│  └─────────────────────────────────────────────────────────────────────────────┤
└─────────────────────────────────────────────────────────────────────────────────┘
```

## System Deployment Architecture

```text
Deployment Options:

┌─────────────────────────────────────────────────────────────────────────────────┐
│                         Development Environment                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────────┤
│  │ Local Machine                                                               │
│  │                                                                             │
│  │ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────┤
│  │ │  AuraTrade  │ │   Ollama    │ │   Paper     │ │   Local     │ │  Demo   ││
│  │ │  Platform   │ │   (Local    │ │  Trading    │ │    Data     │ │  Mode   ││
│  │ │             │ │    LLM)     │ │             │ │   Storage   │ │         ││
│  │ │ • All       │ │             │ │ • No Real   │ │             │ │ • No    ││
│  │ │   Agents    │ │ • Free      │ │   Money     │ │ • JSON      │ │   API   ││
│  │ │ • Risk Mgmt │ │ • Private   │ │ • Safe      │ │ • Memory    │ │   Keys  ││
│  │ │ • Testing   │ │ • Fast      │ │ • Testing   │ │ • Cache     │ │ • Learn ││
│  │ └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘ └─────────┘│
│  └─────────────────────────────────────────────────────────────────────────────┤
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                         Production Environment                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────────┤
│  │ Cloud Infrastructure                                                        │
│  │                                                                             │
│  │ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────┤
│  │ │ Application │ │  Database   │ │    LLM      │ │   Broker    │ │ Monitor ││
│  │ │   Server    │ │   Layer     │ │  Services   │ │    APIs     │ │  & Logs ││
│  │ │             │ │             │ │             │ │             │ │         ││
│  │ │ • Docker    │ │ • InfluxDB  │ │ • OpenAI    │ │ • Alpaca    │ │ • Grafana││
│  │ │ • K8s       │ │ • Redis     │ │ • Anthropic │ │ • Live      │ │ • Alerts ││
│  │ │ • Scale     │ │ • MongoDB   │ │ • Fallback  │ │ • Real $    │ │ • Backup ││
│  │ │ • HA        │ │ • Backup    │ │ • LoadBal   │ │ • Fast      │ │ • Audit  ││
│  │ └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘ └─────────┘│
│  └─────────────────────────────────────────────────────────────────────────────┤
└─────────────────────────────────────────────────────────────────────────────────┘
```