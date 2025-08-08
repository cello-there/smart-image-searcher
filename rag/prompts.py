# rag/prompts.py

# Prompt templates & JSON schemas for LLM-only behavior (no hard-coded rules)

CLARITY_PROMPT = (
    "You are a retrieval assistant. Decide if the user's query requires clarification before image search. "
    "Return strictly JSON.\n\n"
    "Query: {query}\n"
    "Known memory facts (may be empty): {memory}\n\n"
    "Respond with: {{\"ambiguous\": true|false, \"reasons\": [\"...\"]}}"
)

CLARIFY_QS_PROMPT = (
    "You are a retrieval assistant. Given a possibly ambiguous query and known memory facts, "
    "ask up to TWO short clarifying questions that would best disambiguate the request. "
    "Return strictly JSON as {{\"questions\": [\"...\", \"...\"]}} with at most two items.\n\n"
    "Query: {query}\n"
    "Known memory facts: {memory}"
)

INCORPORATE_PROMPT = (
    "Rewrite the user's query using the provided answers to clarification questions. "
    "Keep it concise and specific for vector retrieval. Return JSON {{\"rewritten_query\": \"...\"}}.\n\n"
    "Original query: {query}\n"
    "Answers: {answers}"
)

AUGMENT_PROMPT = (
    "Expand the user's query for CLIP/FAISS image retrieval. Normalize colloquialisms to canonical tags, "
    "include only likely synonyms, and prefer nouns and attributes. Use any relevant memory context.\n\n"
    "Return strictly JSON: {{\"expanded_query\": \"...\", \"terms\": [\"...\"], \"entities\": [\"...\"], \"filters\": {{}}}}\n\n"
    "Query: {query}\n"
    "Context: {context}"
)

CLARITY_SCHEMA = {
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["ambiguous"],
  "properties": {
    "ambiguous": {"type": "boolean"},
    "reasons": {"type": "array", "items": {"type": "string"}}
  },
  "additionalProperties": True
}

CLARIFY_QS_SCHEMA = {
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["questions"],
  "properties": {
    "questions": {"type": "array", "maxItems": 2, "items": {"type": "string"}}
  },
  "additionalProperties": False
}

INCORPORATE_SCHEMA = {
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["rewritten_query"],
  "properties": {
    "rewritten_query": {"type": "string"}
  },
  "additionalProperties": False
}

AUGMENT_SCHEMA = {
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["expanded_query", "terms", "entities"],
  "properties": {
    "expanded_query": {"type": "string"},
    "terms": {"type": "array", "items": {"type": "string"}},
    "entities": {"type": "array", "items": {"type": "string"}},
    "filters": {"type": "object", "additionalProperties": {"type": "string"}}
  },
  "additionalProperties": False
}

MEMORY_DOCS_SCHEMA = {
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["docs"],
  "properties": {
    "docs": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "type": {"type": "string"},
          "key": {"type": "string"},
          "name": {"type": "string"},
          "aliases": {"type": "array", "items": {"type": "string"}},
          "value": {"type": "string"},
          "tags": {"type": "array", "items": {"type": "string"}},
          "source": {"type": "string"}
        },
        "additionalProperties": True
      }
    }
  },
  "additionalProperties": False
}

ANSWERS_TO_MEMORY_PROMPT = (
  "Convert these clarifier answers into memory documents for RAG. "
  "Prefer entity/preference/alias/context types when appropriate. Return strictly JSON {{\"docs\": [ ... ]}}.\n\n"
  "Answers: {answers}"
)
