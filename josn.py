import json

subsistemas = {
    "Sistema de Refrigeración": [
        "RPM", "PT401", "TE401", "TE402", "PT471", "TE471",
        "TE600 - Carga", "TE600 - Aire entrada al turbo"
    ],
    "Sistema de Combustible": [
        "RPM", "PT101", "TE101"
    ],
    "Sistema de Lubricación": [
        "RPM", "PT201", "TE201", "TE202", "TE272", "PDT243",
        "PT271", "XX001", "XX002", "XX012"
    ],
    "Temperatura de Gases de Escape": [
        "RPM", "TE511", "TE517", "TE5011A"
    ]
}

with open("subsistemas.json", "w", encoding="utf-8") as f:
    json.dump(subsistemas, f, indent=2, ensure_ascii=False)
