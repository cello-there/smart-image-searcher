# rag/prompts.py
# Prompt templates & JSON schemas

# ---------- Prompt templates ----------

CLARITY_PROMPT = (
    "You are a retrieval assistant. Decide if the user's query requires clarification before image search. "
    "Return strictly JSON.\n\n"
    "Query: {query}\n"
    "Known memory facts (may be empty): {memory}\n\n"
    "Respond with: {{\"ambiguous\": true|false, \"reasons\": [\"...\"]}}"
)

CLARIFY_QS_PROMPT = (
    "You are a retrieval assistant. Given a possibly ambiguous query and known context, "
    "ask up to TWO short clarifying questions ONLY if the request is still ambiguous.\n"
    "Do NOT ask about fields that are already known in memory or the provided context "
    "(including ephemeral entities just supplied by the user). "
    "Avoid asking about 'which person' if the entity is an inanimate object. "
    "Ask ONLY about missing fields that would materially change retrieval (e.g., which Steve, timeframe, subject type).\n"
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
    "You expand a user query for CLIP/FAISS image retrieval.\n"
    "Rules:\n"
    "  • Output a SINGLE compact phrase (no lists, no 'or').\n"
    "  • Prefer nouns/attributes; include helpful descriptors from memory (color, breed, etc.).\n"
    "  • Use ONLY the entity names/aliases that the user actually typed (do not add other aliases).\n"
    "  • Normalize synonyms like 'images/pictures/photos' to just 'photos'.\n"
    "  • No punctuation lists like 'A, B, C' unless they are attributes for the SAME subject.\n"
    "  • Do NOT emit schema labels/prefixes like 'person:', 'subject:', 'category:'.\n"
    "  • Do NOT quote or bracket names.\n"
    "Return strictly JSON: {{\"expanded_query\":\"...\", \"terms\":[\"...\"], \"entities\":[\"...\"], \"filters\": {{}}}}\n\n"
    "Query: {query}\n"
    "Context (relevant memory): {context}\n"
)

ANSWERS_TO_MEMORY_PROMPT = (
    "Convert these clarifier answers into memory documents for RAG. "
    "Prefer entity/preference/alias/context types when appropriate. "
    "Return strictly JSON as {{\"docs\": [ ... ]}}.\n\n"
    "Answers: {answers}"
)

EXTRACT_ENTITIES_PROMPT = (
    "From the user's query and the known memory, extract any NEW entities the user likely refers to "
    "(pets, people, named objects) that are not already in memory.\n"
    "CRITICAL: Do NOT return generic/common nouns or media words. Examples to AVOID: "
    "'cat', 'cats', 'dog', 'dogs', 'pet', 'pets', 'animal', 'animals', "
    "'person', 'people', 'photos', 'photo', 'pictures', 'images', 'someone', 'something', 'object', 'objects'.\n"
    "Only return specific names/nicknames or clearly named objects (e.g., 'Steve', 'Jason', 'Roomba 694').\n\n"
    "Return strictly JSON as {{\"entities\": [{{\"name\": \"...\", \"category\": \"...\", \"kind\": null}}]}}.\n\n"
    "Query: {query}\n"
    "Known memory: {memory}"
)

ENRICH_ENTITY_PROMPT = (
    "We are saving a new entity to memory. Propose up to THREE short, targeted questions to fill useful fields "
    "(aliases, kind/species, key attributes like color/breed). Ask ONLY about fields that are missing or uncertain "
    "from the name/category/kind/context provided. Do NOT ask about aliases if an alias list was already provided. "
    "Do NOT ask 'which person' if the entity appears to be an object/animal.\n\n"
    "Then propose a small draft entity JSON.\n\n"
    "Return strictly JSON as {{\"questions\":[{{\"q\":\"...\",\"key\":\"aliases|kind|attributes.color|attributes.breed|attributes.note\"}}],"
    "\"entity\":{{\"category\":\"...\",\"kind\":\"...\",\"aliases\":[\"...\"],\"attributes\":{{}},\"description\":\"...\"}}}}\n\n"
    "Name: {name}\n"
    "Category (may be empty): {category}\n"
    "Kind/species (may be empty): {kind}\n"
    "Memory context: {memory}\n"
    "Original query: {query}"
)

PROPOSE_ALIAS_PROMPT = (
    "The user query might contain a nickname or short form for a known entity. "
    "Given the query and a flat list of known names/aliases, propose links.\n\n"
    "Return strictly JSON as {{\"links\": [{{\"entity\": \"<existing name>\", \"alias\": \"<nickname in query>\", \"confidence\": 0.0}}]}}.\n\n"
    "Query: {query}\n"
    "Known names/aliases: {names}\n"
    "Memory summary: {memory}"
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
                    "type": {"type": "string", "enum": ["entity", "preference", "context"]},
                    "category": {"type": "string", "enum": ["pet", "person", "food", "object", "place", "event", "trip", "thing"]},
                    "name": {"type": "string"},
                    "kind": {"type": "string"},
                    "aliases": {"type": "array", "items": {"type": "string"}},
                    "attributes": {"type": "object", "additionalProperties": {"type": "string"}},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "description": {"type": "string"},
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
                    "kind": {"type": ["string", "null"]}
                },
                "additionalProperties": True
            }
        }
    },
    "additionalProperties": False
}

ENRICH_ENTITY_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": ["questions", "entity"],
    "properties": {
        "questions": {
            "type": "array",
            "maxItems": 3,
            "items": {
                "type": "object",
                "required": ["q", "key"],
                "properties": {
                    "q": {"type": "string"},
                    "key": {"type": "string"}
                },
                "additionalProperties": False
            }
        },
        "entity": {
            "type": "object",
            "properties": {
                "category": {"type": "string"},
                "kind": {"type": "string"},
                "aliases": {"type": "array", "items": {"type": "string"}},
                "attributes": {"type": "object"},
                "tags": {"type": "array", "items": {"type": "string"}},
                "description": {"type": "string"}
            },
            "additionalProperties": True
        }
    },
    "additionalProperties": False
}

PROPOSE_ALIAS_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": ["links"],
    "properties": {
        "links": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["entity", "alias"],
                "properties": {
                    "entity": {"type": "string"},
                    "alias": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0}
                },
                "additionalProperties": False
            }
        }
    },
    "additionalProperties": False
}
