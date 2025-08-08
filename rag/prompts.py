# rag/prompts.py

# ---------- Prompt templates ----------
CLARITY_PROMPT = (
    "You are a retrieval assistant. Decide if the user's query requires clarification before image search. "
    "Return strictly JSON.\n\n"
    "Query: {query}\n"
    "Known memory facts (may be empty): {memory}\n\n"
    "Respond with: {{\"ambiguous\": true|false, \"reasons\": [\"...\"]}}"
)

CLARIFY_QS_PROMPT = (
    "You are a retrieval assistant. Given a possibly ambiguous query and known memory facts, "
    "ask up to TWO short clarifying questions that best disambiguate the request.\n"
    "IMPORTANT: Do NOT suggest specific people/pets/objects from memory unless the user already mentioned them. "
    "Keep questions generic.\n"
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
    "Expand the user's query for CLIP/FAISS image retrieval. Normalize colloquialisms to canonical tags; "
    "prefer concise attributes and a single concrete target. "
    "Do NOT enumerate alternatives with 'or' or long comma lists—produce one specific rewritten_query.\n\n"
    "Return strictly JSON: {{\"expanded_query\": \"...\", \"terms\": [\"...\"], \"entities\": [\"...\"], \"filters\": {{}}}}\n\n"
    "Query: {query}\n"
    "Context: {context}"
)


ANSWERS_TO_MEMORY_PROMPT = (
  "Convert these clarifier answers into memory documents for RAG. "
  "Prefer entity/preference/alias/context types when appropriate. "
  "Return strictly JSON as {{\"docs\": [ ... ]}}."
  "\n\nAnswers: {answers}"
)

# ---------- JSON Schemas ----------
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
    "properties": {"rewritten_query": {"type": "string"}},
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
        "required": ["type", "source", "persistent"],
        "properties": {
          "type": {"type": "string", "enum": ["entity","preference","context"]},
          "category": {"type": "string", "enum": ["pet","person","food","object","place","event","trip","thing"]},
          "name": {"type": "string"},
          "kind": {"type": "string"},
          "aliases": {"type": "array", "items": {"type": "string"}},
          "attributes": {"type": "object", "additionalProperties": {"type": "string"}},  # NEW
          "tags": {"type": "array", "items": {"type": "string"}},                        # NEW
          "description": {"type": "string"},                                             # NEW
          "value": {"type": "string"},
          "user_owned": {"type": "boolean"},
          "persistent": {"type": "boolean"},
          "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
          "source": {"type": "string"}
        },
        "additionalProperties": False
      }
    }
  },
  "additionalProperties": False
}

EXTRACT_ENTITIES_PROMPT = (
    "Extract named ENTITIES from the user's search query that could refer to a user-specific thing "
    "(pet/person/food/object/place/event/trip/thing). Keep it minimal and don't guess outside the query. "
    "Return strictly JSON as {{\"entities\": [{{\"name\":\"...\",\"category\":\"pet|person|food|object|place|event|trip|thing\",\"kind\":\"cat|dog|bread|...\"}}]}}."
    "\n\nQuery: {query}"
    "\nKnown entity names (for reference, may be empty): {known_names}"
)

EXTRACT_ENTITIES_SCHEMA = {
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["entities"],
  "properties": {
    "entities": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["name"],
        "properties": {
          "name": {"type": "string"},
          "category": {"type": "string"},
          "kind": {"type": "string"}
        },
        "additionalProperties": False
      }
    }
  },
  "additionalProperties": False
}

ENRICH_ENTITY_PROMPT = (
    "Given a user query and a candidate entity (name/category/kind may be partially known), "
    "propose a minimal structured ENTITY doc and up to THREE targeted questions that will best "
    "disambiguate or describe it (e.g., species/kind, color, aliases). "
    "Use ONLY the information explicit in the query; do not invent facts. "
    "Return strictly JSON as {{"
    "  \"entity\": {{"
    "    \"category\": \"pet|person|food|object|place|event|trip|thing\" | null,"
    "    \"kind\": \"cat|dog|bread|...\" | null,"
    "    \"aliases\": [string],"
    "    \"attributes\": {{string: string}},"
    "    \"tags\": [string],"
    "    \"description\": string"
    "  }},"
    "  \"questions\": ["
    "     {{\"q\": string, \"key\": \"kind\" | \"aliases\" | \"attributes.color\" | \"attributes.breed\" | \"attributes.note\"}}"
    "  ]"
    "}}\n\n"
    "Query: {query}\n"
    "Candidate name: {name}\n"
    "Candidate category (may be null): {category}\n"
    "Candidate kind (may be null): {kind}\n"
    "Known entities for reference (names only): {known_names}"
)

ENRICH_ENTITY_SCHEMA = {
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["entity", "questions"],
  "properties": {
    "entity": {
      "type": "object",
      "required": ["aliases", "attributes", "tags", "description"],
      "properties": {
        "category": {"type": ["string","null"]},
        "kind": {"type": ["string","null"]},
        "aliases": {"type": "array", "items": {"type": "string"}},
        "attributes": {"type": "object", "additionalProperties": {"type": "string"}},
        "tags": {"type": "array", "items": {"type": "string"}},
        "description": {"type": "string"}
      },
      "additionalProperties": False
    },
    "questions": {
      "type": "array",
      "maxItems": 3,
      "items": {
        "type": "object",
        "required": ["q","key"],
        "properties": {
          "q": {"type": "string"},
          "key": {"type": "string"}  # e.g. "kind", "aliases", "attributes.color"
        },
        "additionalProperties": False
      }
    }
  },
  "additionalProperties": False
}

PROPOSE_ALIAS_PROMPT = (
    "You link nicknames to known entities. Consider whether the query uses a nickname of any known entity.\n"
    "Query: {query}\n"
    "Known entities (name, kind, aliases): {entities}\n\n"
    "Return JSON strictly as {{\"links\":[{{\"entity\":\"<existing name>\",\"alias\":\"<new alias>\",\"confidence\":0.0}}]}}. "
    "Only include plausible links; max 2."
)

PROPOSE_ALIAS_SCHEMA = {
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["links"],
  "properties": {
    "links": {
      "type": "array",
      "maxItems": 2,
      "items": {
        "type": "object",
        "required": ["entity", "alias", "confidence"],
        "properties": {
          "entity": {"type": "string"},
          "alias": {"type": "string"},
          "confidence": {"type": "number", "minimum": 0, "maximum": 1}
        },
        "additionalProperties": False
      }
    }
  },
  "additionalProperties": False
}