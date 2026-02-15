"""
MAUDE Auto-Router — Conversation-driven pre-routing for subagents.

Analyzes incoming messages BEFORE the main chat loop using fast keyword
matching and pattern recognition (no LLM call) to route to the optimal
subagent or processing strategy.

This eliminates the tool-loop iteration that occurs when the LLM has to
decide which subagent tool to call, saving a full round-trip per routed
message.

Usage:
    from auto_router import route_message

    decision = route_message("write a Python function to sort a list")
    # decision.intent == "code"
    # decision.subagent == "code"
    # decision.confidence ~= 0.8
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RoutingDecision:
    """Result of message analysis."""
    intent: str                                     # Detected intent category
    confidence: float                               # 0.0-1.0 confidence score
    subagent: Optional[str] = None                  # Which subagent to route to (None = main LLM)
    prefer_cloud: bool = False                      # Whether to prefer cloud for this request
    context_hints: Optional[Dict] = field(          # Extra context to inject
        default_factory=lambda: None
    )

    def __repr__(self) -> str:
        cloud = " [cloud]" if self.prefer_cloud else ""
        agent = self.subagent or "main"
        hints = f" hints={self.context_hints}" if self.context_hints else ""
        return (
            f"RoutingDecision(intent={self.intent!r}, confidence={self.confidence:.2f}, "
            f"agent={agent}{cloud}{hints})"
        )


@dataclass
class _WeightedPattern:
    """A pattern with an associated weight for scoring."""
    pattern: re.Pattern
    weight: float
    hint_key: Optional[str] = None      # Optional key to set in context_hints
    hint_value: Optional[str] = None    # Value for the hint


# ---------------------------------------------------------------------------
# Intent -> subagent mapping
# ---------------------------------------------------------------------------

_INTENT_TO_SUBAGENT: Dict[str, Optional[str]] = {
    "code":         "code",
    "search":       "search",
    "vision":       "vision",
    "writing":      "writer",
    "reasoning":    "reasoning",
    "social_media": None,       # Flagged but handled by skill system, not a subagent
    "browser":      None,       # Flagged but handled by browser automation
    "general":      None,       # Main LLM
}

_INTENT_PREFER_CLOUD: Dict[str, bool] = {
    "code":         False,      # Local codestral is good
    "search":       True,       # Search requires cloud (Grok/Gemini)
    "vision":       False,      # Local LLaVA first
    "writing":      False,      # Local gemma-writer first
    "reasoning":    True,       # Complex reasoning benefits from Claude Opus / o1
    "social_media": True,       # Needs API access
    "browser":      False,      # Local browser automation
    "general":      False,      # Local nemotron
}

# Confidence threshold — below this, route to general
CONFIDENCE_THRESHOLD = 0.3

# History bias factor — how much recent conversation context influences routing
HISTORY_BIAS_WEIGHT = 0.15


# ---------------------------------------------------------------------------
# Pattern definitions
# ---------------------------------------------------------------------------

def _compile(pattern: str, weight: float,
             hint_key: str = None, hint_value: str = None) -> _WeightedPattern:
    """Compile a regex pattern with weight and optional hint."""
    return _WeightedPattern(
        pattern=re.compile(pattern, re.IGNORECASE),
        weight=weight,
        hint_key=hint_key,
        hint_value=hint_value,
    )


def _build_all_patterns() -> Dict[str, List[_WeightedPattern]]:
    """Build the complete pattern registry for all intent categories."""

    patterns: Dict[str, List[_WeightedPattern]] = {}

    # -----------------------------------------------------------------------
    # CODE — Programming, debugging, refactoring, code review
    # -----------------------------------------------------------------------
    patterns["code"] = [
        # Explicit code requests (high weight)
        _compile(r"\b(write|create|generate|build|implement)\b.{0,20}\b(function|class|method|module|script|program|api|endpoint|handler|middleware|decorator|component)\b", 0.9),
        _compile(r"\b(fix|debug|solve|troubleshoot|diagnose)\b.{0,20}\b(bug|error|issue|crash|exception|traceback|segfault|segmentation)\b", 0.9),
        _compile(r"\brefactor\b", 0.85),
        _compile(r"\bcode\s*review\b", 0.9),
        _compile(r"\b(pull\s*request|PR|merge\s*request)\b.{0,15}\b(review|comment|feedback)\b", 0.8),

        # Programming language mentions (medium-high weight)
        _compile(r"\b(python|javascript|typescript|java|c\+\+|rust|go(lang)?|ruby|swift|kotlin|scala|haskell|elixir|clojure|dart|php|perl|lua|zig|nim)\b", 0.5, "language"),
        _compile(r"\b(react|vue|angular|svelte|nextjs|next\.js|django|flask|fastapi|express|spring|rails|laravel)\b", 0.55, "framework"),
        _compile(r"\b(html|css|scss|sass|sql|graphql|grpc|protobuf|yaml|json|xml|toml)\b", 0.4),

        # Code-related actions
        _compile(r"\b(optimize|refactor|clean\s*up|simplify|improve)\b.{0,15}\b(code|function|class|algorithm|logic|performance)\b", 0.85),
        _compile(r"\b(add|implement|create)\b.{0,15}\b(unit\s*test|test\s*case|integration\s*test|e2e\s*test|mock|fixture|test)\b", 0.8),
        _compile(r"\b(how\s*(do|can|to|does))\b.{0,15}\b(import|install|require|include|use|call|invoke|initialize)\b", 0.5),

        # Code patterns and concepts
        _compile(r"\bregex\b|\bregular\s*expression\b|\bregexp\b", 0.9),
        _compile(r"\b(algorithm|data\s*structure|linked\s*list|binary\s*tree|hash\s*map|hash\s*table|graph\s*algorithm|sort(ing)?|search(ing)?)\b", 0.6),
        _compile(r"\b(async|await|promise|callback|thread|mutex|semaphore|coroutine|concurrency|parallelism)\b", 0.55),
        _compile(r"\b(docker(file)?|kubernetes|k8s|helm|terraform|ansible|ci\s*/?\s*cd|github\s*actions?|jenkins|pipeline)\b", 0.5, "devops"),
        _compile(r"\b(git|commit|branch|merge|rebase|cherry-pick|stash)\b", 0.45),
        _compile(r"\b(database|sql|query|migration|schema|orm|model|table|index|join|foreign\s*key)\b", 0.5),
        _compile(r"\b(api|rest|endpoint|route|middleware|controller|service|repository)\b", 0.45),

        # Syntax and error patterns
        _compile(r"\b(syntax\s*error|type\s*error|name\s*error|import\s*error|attribute\s*error|value\s*error|runtime\s*error|compilation\s*error|linker\s*error)\b", 0.85),
        _compile(r"\b(stack\s*trace|traceback|exception|error\s*message|undefined|null\s*pointer|nil|NoneType)\b", 0.7),
        _compile(r"\b(variable|parameter|argument|return\s*type|type\s*hint|annotation|generic|interface|abstract|inheritance|polymorphism)\b", 0.45),

        # Code blocks in message
        _compile(r"```[\w]*\n", 0.7),
        _compile(r"\bdef\s+\w+\s*\(", 0.75),
        _compile(r"\bclass\s+\w+[\s:(]", 0.7),
        _compile(r"\bimport\s+\w+", 0.5),
        _compile(r"\bfrom\s+\w+\s+import\b", 0.6),
        _compile(r"function\s+\w+\s*\(", 0.7),
        _compile(r"const\s+\w+\s*=\s*(async\s+)?\(", 0.65),

        # Package/dependency management
        _compile(r"\b(pip|npm|yarn|cargo|gem|maven|gradle|composer|nuget|brew)\s+(install|add|update|upgrade|remove)\b", 0.7),
        _compile(r"\b(requirements\.txt|package\.json|Cargo\.toml|Gemfile|pom\.xml|build\.gradle)\b", 0.6),
    ]

    # -----------------------------------------------------------------------
    # SEARCH — Current events, real-time data, web lookup
    # -----------------------------------------------------------------------
    patterns["search"] = [
        # Explicit search requests
        _compile(r"\b(search|look\s*up|google|find\s*(out|me|information|info))\b.{0,20}\b(about|for|on|regarding)\b", 0.9),
        _compile(r"\b(what('s| is| are)|who('s| is| are)|when (is|was|did|does|will)|where (is|are|was|did|can))\b.{0,30}\b(latest|current|recent|now|today|tonight|this\s*(week|month|year))\b", 0.9),

        # Current events and news
        _compile(r"\b(news|headline|breaking|current\s*events?|what('s| is)\s*happening)\b", 0.85),
        _compile(r"\b(latest|recent|new|upcoming|current)\b.{0,20}\b(update|release|version|announcement|report)\b", 0.7),
        _compile(r"\b(today|tonight|yesterday|this\s*week|this\s*month|right\s*now|currently)\b.{0,15}\b(weather|temperature|forecast|traffic|score|result|price|rate)\b", 0.9),

        # Real-time data
        _compile(r"\b(stock|share)\s*(price|market|ticker|symbol|quote)\b", 0.9, "data_type", "stocks"),
        _compile(r"\b(bitcoin|ethereum|btc|eth|crypto(currency)?|coin)\s*(price|value|market|cap)\b", 0.9, "data_type", "crypto"),
        _compile(r"\$[A-Z]{1,5}\b", 0.6, "data_type", "stocks"),  # $AAPL, $TSLA, etc.
        _compile(r"\b(exchange\s*rate|forex|usd|eur|gbp|jpy)\b.{0,15}\b(to|rate|convert|price)\b", 0.8, "data_type", "forex"),
        _compile(r"\b(score|result|standing|ranking|fixture)\b.{0,15}\b(game|match|nfl|nba|mlb|nhl|premier\s*league|soccer|football|basketball|baseball|hockey)\b", 0.9, "data_type", "sports"),
        # Sports questions — "who won the NBA game" etc.
        _compile(r"\b(who\s*(won|lost|beat|scored|plays?|playing))\b.{0,25}\b(game|match|nfl|nba|mlb|nhl|premier\s*league|series|cup|championship|tournament|super\s*bowl|world\s*series)\b", 0.9, "data_type", "sports"),
        _compile(r"\b(nfl|nba|mlb|nhl|premier\s*league|champions\s*league|world\s*cup|super\s*bowl|march\s*madness)\b.{0,20}\b(score|result|winner|game|match|schedule|standing|playoff)\b", 0.85, "data_type", "sports"),
        _compile(r"\b(last\s*night|yesterday|tonight|this\s*weekend)\b.{0,15}\b(game|match|fight|bout|race)\b", 0.75, "data_type", "sports"),

        # Information lookup
        _compile(r"\bhow\s+(much|many|long|far|old|tall|big)\b.{0,10}\b(is|are|does|did|was|were)\b", 0.4),
        _compile(r"\b(who\s+(is|was|are|were|won|invented|discovered|founded|created))\b", 0.55),
        _compile(r"\b(when\s+(is|was|did|does|will))\b", 0.45),
        _compile(r"\b(population|gdp|capital|president|ceo|founder)\s*(of|for)\b", 0.6),

        # Explicit real-time indicators
        _compile(r"\b(real[\s-]*time|live|up[\s-]*to[\s-]*date|current)\b", 0.5),
        _compile(r"\b(weather|forecast|temperature)\b.{0,15}\b(in|for|at|today|tomorrow|this\s*week)\b", 0.85, "data_type", "weather"),
        _compile(r"\b(flight|flights?)\b.{0,15}\b(status|delayed|from|to|price|cheap|book)\b", 0.7, "data_type", "travel"),
        _compile(r"\b(restaurant|hotel|store|shop)\b.{0,15}\b(near|nearby|close|around|in|open|hours|review)\b", 0.65, "data_type", "local"),

        # Events and schedules
        _compile(r"\b(schedule|event|concert|show|movie|film)\b.{0,15}\b(today|tonight|this\s*week|near|showing|playing|time)\b", 0.7),
        _compile(r"\b(what\s*time|when)\b.{0,20}\b(game|match|show|movie|concert|event|flight)\b.{0,15}\b(start|begin|on|tonight|today|tomorrow)\b", 0.85, "data_type", "events"),
        _compile(r"\b(game|match)\b.{0,15}\b(start|begin|time|tonight|today|tomorrow|this\s*week)\b", 0.6, "data_type", "sports"),
        _compile(r"\b(election|vote|poll|ballot|primary|debate)\b", 0.6, "data_type", "politics"),
        _compile(r"\b(release\s*date|launch\s*date|premiere|coming\s*out|drops?)\b.{0,15}\b(when|date)\b", 0.6),
    ]

    # -----------------------------------------------------------------------
    # VISION — Image analysis, screenshots, visual understanding
    # -----------------------------------------------------------------------
    patterns["vision"] = [
        # Explicit image requests
        _compile(r"\b(look\s*at|analyze|describe|examine|inspect|read|scan|check)\b.{0,15}\b(this\s*)?(image|picture|photo|screenshot|screen\s*cap|snap|diagram|chart|graph|figure|illustration)\b", 0.95),
        _compile(r"\b(what('s| is|'re| are))\b.{0,10}\b(in\s*)?(this\s*)?(image|picture|photo|screenshot)\b", 0.9),
        _compile(r"\b(can\s*you\s*see|do\s*you\s*see|what\s*do\s*you\s*see)\b", 0.7),
        _compile(r"\b(ocr|extract\s*text|read\s*text)\b.{0,15}\b(from|in)\b", 0.85),

        # Image file paths
        _compile(r"\.(png|jpg|jpeg|gif|bmp|tiff|tif|webp|svg|heic|heif|avif)\b", 0.9, "has_image", "true"),
        _compile(r"(https?://\S+\.(png|jpg|jpeg|gif|bmp|webp|svg))", 0.85, "has_image_url", "true"),
        _compile(r"\b(image|img|photo|pic|screenshot)[\s_-]*(file|path|url|link)\b", 0.7),

        # Visual analysis tasks
        _compile(r"\b(ui|ux|interface|layout|design|mockup|wireframe)\b.{0,15}\b(feedback|review|analyze|issue|problem|bug)\b", 0.7),
        _compile(r"\b(this\s*error|error\s*message)\b.{0,10}\b(screen|shows?|says?|displays?)\b", 0.6),
        _compile(r"\b(chart|graph|plot|diagram|table|infographic|dashboard)\b.{0,15}\b(shows?|means?|explain|interpret|read|understand)\b", 0.8),

        # Standalone visual content words — when someone says "this diagram" they likely have an image
        _compile(r"\b(this|the)\s*(image|picture|photo|screenshot|diagram|chart|graph|figure)\b", 0.75),
        _compile(r"\b(what\s*does)\b.{0,15}\b(this|the)\b.{0,10}\b(show|display|depict|illustrate|mean|say|represent)\b", 0.6),

        # Attached/uploaded content
        _compile(r"\b(attached|uploaded|here('s| is)|see\s*(the|this))\b.{0,15}\b(image|picture|photo|screenshot|file)\b", 0.85),
        _compile(r"\b(I('m| am)\s*sharing|take\s*a\s*look)\b", 0.4),
    ]

    # -----------------------------------------------------------------------
    # WRITING — Long-form content, documentation, emails, summaries
    # -----------------------------------------------------------------------
    patterns["writing"] = [
        # Explicit writing requests (high weight)
        # Use generous gap (40 chars) to allow qualifiers like "me a professional"
        _compile(r"\b(write|draft|compose|create|author)\b.{0,40}\b(email|e-mail|letter|memo|message|note|correspondence)\b", 0.9, "writing_type", "email"),
        _compile(r"\b(write|draft|compose|create)\b.{0,40}\b(blog\s*post|article|essay|paper|report|whitepaper|case\s*study)\b", 0.9, "writing_type", "article"),
        _compile(r"\b(write|draft|compose|create)\b.{0,40}\b(documentation|docs?|readme|guide|tutorial|manual|handbook|wiki)\b", 0.85, "writing_type", "documentation"),
        _compile(r"\b(write|draft|compose|create)\b.{0,40}\b(proposal|pitch|brief|plan|specification|spec|rfc)\b", 0.85, "writing_type", "proposal"),
        _compile(r"\b(write|draft|compose|create)\b.{0,40}\b(story|poem|narrative|fiction|screenplay|script|dialogue|monologue)\b", 0.85, "writing_type", "creative"),
        _compile(r"\b(write|draft|compose|create)\b.{0,40}\b(resume|cv|cover\s*letter|bio|biography|about\s*me|introduction)\b", 0.85, "writing_type", "professional"),

        # Summarization
        _compile(r"\bsummar(ize|y)\b", 0.8, "writing_type", "summary"),
        _compile(r"\b(tldr|tl;dr|too\s*long|brief\s*overview|key\s*points?|main\s*points?|takeaways?|highlights?)\b", 0.75, "writing_type", "summary"),
        _compile(r"\b(condense|shorten|abridge|distill|boil\s*down|cliff\s*notes)\b", 0.7, "writing_type", "summary"),

        # Editing and rewriting
        _compile(r"\b(rewrite|rephrase|paraphrase|improve|polish|revise|edit|proofread|copyedit)\b.{0,15}\b(this|the|my|following|text|paragraph|sentence|draft|document|article)\b", 0.8),
        _compile(r"\b(make\s*(it|this))\b.{0,15}\b(more\s*)?(formal|informal|professional|casual|concise|clear|readable|engaging|persuasive|friendly|shorter|longer)\b", 0.75),
        _compile(r"\b(tone|voice|style)\b.{0,10}\b(change|adjust|modify|more|less)\b", 0.7),

        # Content structure
        _compile(r"\b(outline|structure|organize|format|layout)\b.{0,15}\b(for|of|the|this|my|a)\b.{0,15}\b(document|article|paper|post|report|presentation)\b", 0.75),
        _compile(r"\b(table\s*of\s*contents|section|chapter|heading|bullet\s*points?|numbered\s*list)\b", 0.4),

        # Professional/business writing
        _compile(r"\b(meeting\s*notes?|minutes|agenda|action\s*items?|follow[\s-]*up)\b", 0.65, "writing_type", "meeting"),
        _compile(r"\b(announcement|press\s*release|newsletter|marketing\s*copy|ad\s*copy|tagline|slogan|headline)\b", 0.75, "writing_type", "marketing"),
        _compile(r"\b(business\s*plan|executive\s*summary|mission\s*statement|vision\s*statement)\b", 0.8, "writing_type", "business"),

        # Translation — generous gap to allow "this paragraph to Spanish" etc.
        _compile(r"\b(translate|translation)\b.{0,40}\b(to|into|from)\b.{0,20}\b(english|spanish|french|german|chinese|japanese|korean|italian|portuguese|arabic|hindi|russian)\b", 0.8, "writing_type", "translation"),
        _compile(r"\b(translate|translation)\b.{0,30}\b(to|into|from)\b", 0.65, "writing_type", "translation"),

        # Long-form indicators
        _compile(r"\b(detailed|comprehensive|thorough|in[\s-]*depth|extensive|complete)\b.{0,15}\b(explanation|overview|analysis|description|guide|breakdown)\b", 0.6),
        _compile(r"\b(explain|describe|elaborate)\b.{0,10}\b(in\s*detail|thoroughly|step\s*by\s*step|at\s*length|comprehensively)\b", 0.55),
    ]

    # -----------------------------------------------------------------------
    # REASONING — Complex analysis, planning, comparison, multi-step
    # -----------------------------------------------------------------------
    patterns["reasoning"] = [
        # Explicit reasoning requests
        _compile(r"\b(analyze|analysis|evaluate|assess|critique|appraise)\b.{0,15}\b(the|this|my|these|those|following)\b", 0.75),
        _compile(r"\b(compare|contrast|differentiate|distinguish)\b.{0,15}\b(between|and|versus|vs\.?)\b", 0.85),
        _compile(r"\b(difference|differences)\s*(between|of)\b", 0.8),
        _compile(r"\b(explain|what('s| is))\b.{0,15}\b(the\s*)?(difference|distinction)\b", 0.75),
        _compile(r"\b(pros?\s*(and|&|\/)\s*cons?)\b", 0.9),
        _compile(r"\b(advantages?\s*(and|&|\/)\s*disadvantages?|strengths?\s*(and|&|\/)\s*weaknesses?|benefits?\s*(and|&|\/)\s*drawbacks?)\b", 0.85),
        _compile(r"\b(trade[\s-]*offs?|trade[\s-]*off)\b", 0.7),

        # Planning and strategy
        _compile(r"\b(plan|planning|strategy|strategize|roadmap|game\s*plan|action\s*plan)\b.{0,15}\b(for|to|how|the|my|a)\b", 0.8),
        _compile(r"\b(step[\s-]*by[\s-]*step|step\s*by\s*step|walkthrough|walk\s*me\s*through)\b", 0.6),
        _compile(r"\b(how\s*(should|would|can|do)\s*(I|we|you))\b.{0,20}\b(approach|tackle|handle|solve|address|deal\s*with|manage|navigate|proceed|start)\b", 0.8),
        _compile(r"\b(best\s*(approach|way|method|strategy|practice)|optimal|most\s*effective|most\s*efficient)\b", 0.6),
        _compile(r"\b(decision|decide|choose|choosing|choice|pick|select|which\s*(one|option|approach))\b.{0,10}\b(between|should|is\s*better|makes?\s*sense|would\s*you)\b", 0.85),
        _compile(r"\bshould\s*(I|we)\s*(use|choose|pick|go\s*with|adopt|switch\s*to|migrate\s*to)\b", 0.8),
        _compile(r"\b\w+\s*(or|vs\.?|versus)\s*\w+\b.{0,20}\b(for|which|better|recommend|prefer|choose)\b", 0.7),

        # Complex analysis
        _compile(r"\b(root\s*cause|why\s*(does|did|is|are|was|were|do|would))\b.{0,20}\b(happen|fail|work|occur|exist|matter)\b", 0.7),
        _compile(r"\b(implications?|consequences?|impact|ramifications?|repercussions?|side\s*effects?)\b.{0,10}\b(of|for|on)\b", 0.7),
        _compile(r"\b(what\s*(if|would\s*happen)|hypothetic(al|ally)|scenario|thought\s*experiment)\b", 0.75),
        _compile(r"\b(risk|risks?)\b.{0,10}\b(analysis|assessment|mitigation|management|of|associated|involved)\b", 0.8),
        _compile(r"\b(feasibility|viability|sustainability|scalability)\b", 0.65),

        # Logical reasoning
        _compile(r"\b(logic(al|ally)?|reason(ing)?|deduc(e|tion)|induc(e|tion)|infer(ence)?)\b", 0.6),
        _compile(r"\b(therefore|consequently|thus|hence|implies?|assuming|given\s*that|it\s*follows)\b", 0.35),
        _compile(r"\b(paradox|dilemma|fallacy|contradiction|counter[\s-]*argument|counter[\s-]*example)\b", 0.7),

        # Multi-step / complex tasks
        _compile(r"\b(complex|complicated|intricate|nuanced|multi[\s-]*faceted|multi[\s-]*step|non[\s-]*trivial)\b.{0,15}\b(problem|question|issue|task|challenge|situation)\b", 0.75),
        _compile(r"\b(break\s*(down|apart)|decompose|dissect|unpack)\b.{0,15}\b(this|the|problem|issue|question|concept)\b", 0.7),
        _compile(r"\b(help\s*me\s*(think|understand|figure\s*out|work\s*through|reason))\b", 0.75),

        # Architecture and design decisions
        _compile(r"\b(architect(ure)?|system\s*design|design\s*pattern|design\s*decision)\b.{0,15}\b(for|of|the|should|best|recommend)\b", 0.7),
        _compile(r"\b(when\s*(should|to)\s*(use|choose|pick|prefer))\b.{0,15}\b(over|instead\s*of|vs\.?|versus)\b", 0.75),

        # Math and quantitative reasoning
        _compile(r"\b(calculate|compute|derive|solve|prove|equation|formula|theorem|mathematical)\b", 0.55),
        _compile(r"\b(probability|statistic(s|al)|correlation|regression|distribution|standard\s*deviation|mean|median|variance)\b", 0.5),
    ]

    # -----------------------------------------------------------------------
    # SOCIAL MEDIA — Posts, tweets, social content
    # -----------------------------------------------------------------------
    patterns["social_media"] = [
        _compile(r"\b(post|share|publish)\b.{0,15}\b(to|on)\b.{0,10}\b(twitter|x\.com|linkedin|facebook|instagram|mastodon|bluesky|threads|reddit|tiktok)\b", 0.95),
        _compile(r"\b(tweet|retweet|toot)\b", 0.85),
        _compile(r"\b(linkedin)\b.{0,10}\b(post|update|article|share)\b", 0.9),
        _compile(r"\b(social\s*media)\b.{0,10}\b(post|update|content|strategy|calendar)\b", 0.85),
        _compile(r"\b(hashtag|caption|viral|engagement|followers?|likes?|repost)\b.{0,15}\b(for|on|about|strategy)\b", 0.6),
        _compile(r"\b(instagram)\b.{0,10}\b(story|stories|reel|post|caption)\b", 0.85),
        _compile(r"\b(thread|tweetstorm)\b.{0,10}\b(about|on|for)\b", 0.7),
    ]

    # -----------------------------------------------------------------------
    # BROWSER — Web automation, form filling, navigation
    # -----------------------------------------------------------------------
    patterns["browser"] = [
        _compile(r"\b(go\s*to|navigate\s*to|open|visit|browse)\b.{0,10}\b(the\s*)?(website|webpage|page|site|url|link)\b", 0.9),
        _compile(r"\b(go\s*to|navigate\s*to|open|visit)\b.{0,5}(https?://|www\.)\S+", 0.95),
        _compile(r"\b(fill\s*(out|in)|complete|submit)\b.{0,30}\b(the\s*)?(form|field|input|application|survey)\b", 0.9),
        _compile(r"\b(click|tap|press|select|choose)\b.{0,10}\b(the\s*)?(button|link|menu|dropdown|checkbox|radio|tab|option|element)\b", 0.85),
        _compile(r"\b(scroll|scroll\s*down|scroll\s*to|swipe)\b", 0.6),
        _compile(r"\b(browser|chrome|firefox|selenium|playwright|puppeteer)\b.{0,10}\b(automation|automate|open|launch|use)\b", 0.85),
        _compile(r"\b(scrape|crawl|extract)\b.{0,15}\b(from|the|this)\b.{0,10}\b(website|page|site|url)\b", 0.8),
        _compile(r"\b(log\s*in|login|sign\s*in|authenticate)\b.{0,10}\b(to|on|at)\b.{0,10}\b(the\s*)?(website|site|page|portal|account)\b", 0.85),
        _compile(r"\b(download|save)\b.{0,10}\b(from|the|this)\b.{0,10}\b(website|page|site|link|url)\b", 0.7),
        _compile(r"\b(take\s*a\s*screenshot|screenshot\s*of)\b.{0,10}\b(the\s*)?(website|page|site)\b", 0.8),
    ]

    return patterns


# ---------------------------------------------------------------------------
# Message Router
# ---------------------------------------------------------------------------

class MessageRouter:
    """
    Analyzes messages and routes to optimal processing path.

    Uses weighted keyword/regex patterns scored per intent category.
    Highest scoring intent above the confidence threshold wins.
    Below threshold falls back to "general" (main LLM).
    """

    def __init__(self) -> None:
        self._patterns: Dict[str, List[_WeightedPattern]] = _build_all_patterns()
        # Precompute max possible score per category for normalization
        self._max_scores: Dict[str, float] = {
            intent: sum(p.weight for p in pats)
            for intent, pats in self._patterns.items()
        }
        # Precompute normalization divisors — based on the sum of the top-3
        # weights per category. This means matching 3 strong patterns yields
        # a confidence near 1.0, and a single strong pattern yields ~0.3-0.5.
        self._norm_divisors: Dict[str, float] = {}
        for intent, pats in self._patterns.items():
            top_weights = sorted((p.weight for p in pats), reverse=True)[:3]
            self._norm_divisors[intent] = sum(top_weights) if top_weights else 1.0

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------

    def route(self, message: str, history: List[dict] = None) -> RoutingDecision:
        """
        Analyze a message and return a routing decision.

        Args:
            message: The user's incoming message text.
            history: Optional conversation history (list of dicts with
                     at minimum a "content" key and optionally "role").

        Returns:
            RoutingDecision with intent, confidence, subagent, etc.
        """
        if not message or not message.strip():
            return RoutingDecision(intent="general", confidence=0.0)

        # Score every intent category
        raw_scores: Dict[str, float] = {}
        all_hints: Dict[str, Dict[str, str]] = {}

        for intent, patterns in self._patterns.items():
            score, hints = self._score_intent(message, patterns)
            raw_scores[intent] = score
            if hints:
                all_hints[intent] = hints

        # Normalize scores to 0.0-1.0 range
        # We use a practical normalization: hitting 2-3 strong patterns should
        # yield high confidence. The divisor is the mean of the top-3 pattern
        # weights for each category, times a scaling factor. This prevents
        # categories with many patterns from being penalized relative to
        # categories with few patterns.
        normalized: Dict[str, float] = {}
        for intent, raw in raw_scores.items():
            norm_divisor = self._norm_divisors.get(intent, 1.0)
            normalized[intent] = min(1.0, raw / max(norm_divisor, 0.01))

        # Apply history bias
        if history:
            history_bias = self._compute_history_bias(history)
            for intent, bias in history_bias.items():
                if intent in normalized:
                    normalized[intent] = min(1.0, normalized[intent] + bias)

        # Find the winning intent
        if not normalized:
            return RoutingDecision(intent="general", confidence=0.0)

        best_intent = max(normalized, key=normalized.get)
        best_score = normalized[best_intent]

        # Apply threshold
        if best_score < CONFIDENCE_THRESHOLD:
            return RoutingDecision(intent="general", confidence=best_score)

        # Build context hints
        context_hints = all_hints.get(best_intent)

        # Check for ambiguity — if two intents are very close, note it
        sorted_intents = sorted(normalized.items(), key=lambda x: x[1], reverse=True)
        if len(sorted_intents) >= 2:
            runner_up_intent, runner_up_score = sorted_intents[1]
            if runner_up_score > 0 and (best_score - runner_up_score) < 0.1:
                # Close call — add the runner-up as a hint
                if context_hints is None:
                    context_hints = {}
                context_hints["ambiguous_with"] = runner_up_intent
                context_hints["runner_up_confidence"] = f"{runner_up_score:.2f}"

        # Map to subagent
        subagent = _INTENT_TO_SUBAGENT.get(best_intent)
        prefer_cloud = _INTENT_PREFER_CLOUD.get(best_intent, False)

        # Special case: if the message contains code AND asks to explain,
        # it might be better as reasoning
        if best_intent == "code" and context_hints and "ambiguous_with" in context_hints:
            if context_hints["ambiguous_with"] == "reasoning":
                # Check if the message is asking to explain/understand rather than write
                explain_pattern = re.compile(
                    r"\b(explain|what\s*does|how\s*does|why\s*does|understand|meaning\s*of|purpose\s*of)\b",
                    re.IGNORECASE
                )
                if explain_pattern.search(message):
                    best_intent = "reasoning"
                    best_score = normalized.get("reasoning", best_score)
                    subagent = _INTENT_TO_SUBAGENT.get("reasoning")
                    prefer_cloud = _INTENT_PREFER_CLOUD.get("reasoning", False)

        # Special case: search requests about code/programming topics
        # should stay with search, not get reclassified as code
        if best_intent == "search":
            pass  # Search intent takes priority when detected — user wants live data

        # Special case: very short messages with low confidence
        # are often greetings or simple questions — keep as general
        word_count = len(message.split())
        if word_count <= 3 and best_score < 0.5:
            return RoutingDecision(intent="general", confidence=best_score)

        decision = RoutingDecision(
            intent=best_intent,
            confidence=round(best_score, 3),
            subagent=subagent,
            prefer_cloud=prefer_cloud,
            context_hints=context_hints if context_hints else None,
        )

        log.debug("Auto-route: %s -> %s", message[:80], decision)
        return decision

    # -------------------------------------------------------------------
    # Internal scoring
    # -------------------------------------------------------------------

    def _score_intent(
        self,
        message: str,
        patterns: List[_WeightedPattern],
    ) -> Tuple[float, Dict[str, str]]:
        """
        Score a message against a list of weighted patterns.

        Returns:
            Tuple of (total_score, hints_dict).
        """
        total = 0.0
        hints: Dict[str, str] = {}

        for wp in patterns:
            match = wp.pattern.search(message)
            if match:
                total += wp.weight
                if wp.hint_key and wp.hint_value:
                    hints[wp.hint_key] = wp.hint_value
                elif wp.hint_key and not wp.hint_value:
                    # Use the matched text as the hint value
                    hints[wp.hint_key] = match.group(0).strip().lower()

        return total, hints

    def _compute_history_bias(
        self,
        history: List[dict],
        lookback: int = 5,
    ) -> Dict[str, float]:
        """
        Compute intent bias from recent conversation history.

        If recent messages were about code, we bias toward code, etc.
        Only looks at the last `lookback` messages.

        Returns:
            Dict mapping intent -> bias value (added to normalized score).
        """
        bias: Dict[str, float] = {}
        recent = history[-lookback:] if len(history) > lookback else history

        # Count how many recent messages match each intent
        intent_hits: Dict[str, int] = {}
        for msg in recent:
            content = msg.get("content", "")
            if not content:
                continue
            for intent, patterns in self._patterns.items():
                score, _ = self._score_intent(content, patterns)
                if score > 0:
                    intent_hits[intent] = intent_hits.get(intent, 0) + 1

        # Convert hits to bias (max HISTORY_BIAS_WEIGHT)
        if intent_hits:
            max_hits = max(intent_hits.values())
            for intent, hits in intent_hits.items():
                bias[intent] = (hits / max_hits) * HISTORY_BIAS_WEIGHT

        return bias

    # -------------------------------------------------------------------
    # Introspection
    # -------------------------------------------------------------------

    def get_intent_categories(self) -> List[str]:
        """Return all supported intent category names."""
        return list(self._patterns.keys()) + ["general"]

    def get_pattern_count(self, intent: str) -> int:
        """Return number of patterns registered for an intent."""
        return len(self._patterns.get(intent, []))

    def explain(self, message: str, history: List[dict] = None) -> str:
        """
        Return a human-readable explanation of how a message would be routed.

        Useful for debugging and tuning patterns.
        """
        decision = self.route(message, history)
        lines = [
            f"Message: {message[:100]}{'...' if len(message) > 100 else ''}",
            f"Decision: {decision.intent} (confidence: {decision.confidence:.2f})",
            f"Subagent: {decision.subagent or 'main LLM'}",
            f"Prefer cloud: {decision.prefer_cloud}",
        ]

        if decision.context_hints:
            lines.append(f"Context hints: {decision.context_hints}")

        # Show all scores
        lines.append("\nAll intent scores:")
        raw_scores: Dict[str, float] = {}
        for intent, patterns in self._patterns.items():
            score, _ = self._score_intent(message, patterns)
            norm_divisor = self._norm_divisors.get(intent, 1.0)
            norm = min(1.0, score / max(norm_divisor, 0.01))
            raw_scores[intent] = norm

        for intent, score in sorted(raw_scores.items(), key=lambda x: x[1], reverse=True):
            marker = " <--" if intent == decision.intent else ""
            lines.append(f"  {intent:15} {score:.3f}{marker}")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Singleton and convenience functions
# ---------------------------------------------------------------------------

_router_instance: Optional[MessageRouter] = None


def get_router() -> MessageRouter:
    """Get or create the global MessageRouter singleton."""
    global _router_instance
    if _router_instance is None:
        _router_instance = MessageRouter()
    return _router_instance


def route_message(message: str, history: List[dict] = None) -> RoutingDecision:
    """
    Convenience function — route a message using the global router.

    Args:
        message: The user's incoming message text.
        history: Optional conversation history (list of dicts with
                 at minimum a "content" key).

    Returns:
        RoutingDecision with intent, confidence, subagent, etc.
    """
    return get_router().route(message, history)


# ---------------------------------------------------------------------------
# CLI testing interface
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    router = get_router()

    # Print pattern stats
    print("MAUDE Auto-Router Pattern Statistics")
    print("=" * 50)
    for intent in router.get_intent_categories():
        count = router.get_pattern_count(intent)
        if count > 0:
            print(f"  {intent:15} {count:3} patterns")
    print()

    # Interactive or argument mode
    if len(sys.argv) > 1:
        test_msg = " ".join(sys.argv[1:])
        print(router.explain(test_msg))
    else:
        # Run built-in test cases
        test_cases = [
            # Code
            "write a Python function to parse CSV files",
            "fix this bug in my JavaScript code",
            "refactor the database module to use async/await",
            "what does this regex do: ^[a-zA-Z0-9]+$",
            "```python\ndef foo():\n    pass\n```\nwhy is this failing?",

            # Search
            "what's the current price of Bitcoin?",
            "who won the NBA game last night?",
            "what's happening in Ukraine today?",
            "weather in San Francisco this week",
            "latest Tesla stock price",

            # Vision
            "look at this screenshot and tell me what's wrong",
            "analyze the image at /home/user/chart.png",
            "what does this diagram show?",

            # Writing
            "write me a professional email to decline a meeting",
            "summarize this 10-page document",
            "draft a blog post about machine learning trends",
            "translate this paragraph to Spanish",

            # Reasoning
            "compare React vs Vue for a new project",
            "what are the pros and cons of microservices?",
            "help me plan the architecture for a chat application",
            "should I use PostgreSQL or MongoDB for this use case?",

            # Social media
            "post this to Twitter: excited about the new release!",
            "write a LinkedIn post about our product launch",

            # Browser
            "go to https://github.com and find the trending repos",
            "fill out the registration form on the website",

            # General (should not route)
            "hello",
            "how are you?",
            "thanks!",
            "tell me a joke",
        ]

        for msg in test_cases:
            decision = route_message(msg)
            conf_bar = "#" * int(decision.confidence * 20)
            agent_label = decision.subagent or "main"
            cloud_flag = " [cloud]" if decision.prefer_cloud else ""
            print(f"  [{decision.intent:13}] {conf_bar:20} {decision.confidence:.2f}  -> {agent_label:10}{cloud_flag}  | {msg[:60]}")
