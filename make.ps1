param([Parameter(Position=0)][string]$Task="help")

function Train { python .\train_and_save.py; bentoml models list }
function Build { bentoml build; bentoml containerize energy_api:latest -t energy_api:latest }
function Run {
  docker ps -q --filter "publish=3000" | ForEach-Object { docker rm -f $_ } | Out-Null
  docker rm -f energy_api 2>$null | Out-Null
  docker run -d --name energy_api -p 3000:3000 energy_api:latest
  docker ps --filter "name=energy_api"
}

function Logs  { docker logs -f energy_api }
function Stop  { docker rm -f energy_api }
function Inspect {
  $payload = Get-Content .\request.json -Raw | ConvertFrom-Json
  $json = (@{ payload = $payload } | ConvertTo-Json -Depth 10 -Compress)
  $enc  = New-Object System.Text.UTF8Encoding($false)   # UTF-8 SANS BOM
  $bytes = $enc.GetBytes($json)
  Invoke-RestMethod -Uri "http://localhost:3000/inspect" -Method Post -ContentType "application/json; charset=utf-8" -Body $bytes
}
function Predict {
  $payload = Get-Content .\request.json -Raw | ConvertFrom-Json
  $json = (@{ payload = $payload } | ConvertTo-Json -Depth 10 -Compress)
  $enc  = New-Object System.Text.UTF8Encoding($false)   # UTF-8 SANS BOM
  $bytes = $enc.GetBytes($json)
  Invoke-RestMethod -Uri "http://localhost:3000/predict" -Method Post -ContentType "application/json; charset=utf-8" -Body $bytes
}function Help {
  @"
Usage: .\make.ps1 <task>

Tasks:
  train       — entraîne + sauvegarde le modèle
  build       — bento build + docker image
  run         — lance le conteneur en arrière-plan (port 3000)
  logs        — suit les logs du conteneur
  stop        — stoppe & supprime le conteneur
  inspect     — envoie request.json à /inspect (enveloppe auto sans BOM)
  predict     — envoie request.json à /predict (enveloppe auto sans BOM)
"@
}

switch ($Task) {
  "train"   { Train }
  "build"   { Build }
  "run"     { Run }
  "logs"    { Logs }
  "stop"    { Stop }
  "inspect" { Inspect }
  "predict" { Predict }
  default   { Help }
}
