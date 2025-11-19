# Django ML API

API REST con Django para procesamiento de datos de Machine Learning basada en el dataset NSL-KDD.

## Características

- **Análisis de Datasets**: Obtén información detallada sobre tus datos
- **División de Datos**: Split estratificado en train/validation/test
- **Transformaciones**: Transformadores personalizados de sklearn
  - Eliminación de filas con NaN
  - Escalado robusto de características
  - One-Hot Encoding personalizado

## Endpoints

### Health Check
\`\`\`
GET /api/health/
\`\`\`

### Dataset Info
\`\`\`
POST /api/dataset/info/
Body: {
  "data": [{"col1": "val1", "col2": "val2"}, ...]
}
\`\`\`

### Data Split
\`\`\`
POST /api/dataset/split/
Body: {
  "data": [...],
  "random_state": 42,
  "shuffle": true,
  "stratify_column": "protocol_type"
}
\`\`\`

### Transform Data
\`\`\`
POST /api/dataset/transform/
Body: {
  "data": [...],
  "remove_nan": false,
  "scale_columns": ["duration", "src_bytes"],
  "one_hot_encode": true
}
\`\`\`

## Instalación Local

\`\`\`bash
# Clonar repositorio
git clone <tu-repo>
cd <tu-repo>

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Migraciones
python manage.py migrate

# Ejecutar servidor
python manage.py runserver
\`\`\`

## Deployment

### Vercel (Serverless)

1. Instala Vercel CLI: `npm i -g vercel`
2. Ejecuta: `vercel`
3. Sigue las instrucciones

### Render (Web Service)

1. Conecta tu repositorio de GitHub
2. Configura:
   - Build Command: `pip install -r requirements.txt && python manage.py collectstatic --noinput`
   - Start Command: `gunicorn ml_api.wsgi:application`
3. Añade variables de entorno:
   - `SECRET_KEY`: Tu clave secreta de Django
   - `DEBUG`: False

## Estructura del Proyecto

\`\`\`
.
├── ml_api/              # Configuración Django
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
├── api/                 # App principal
│   ├── views.py         # Endpoints API
│   ├── serializers.py   # Validación de datos
│   ├── transformers.py  # Transformadores sklearn
│   └── utils.py         # Utilidades
├── requirements.txt     # Dependencias Python
├── vercel.json         # Configuración Vercel
└── README.md
\`\`\`

## Tecnologías

- Django 4.2
- Django REST Framework
- scikit-learn
- pandas
- numpy

## Licencia

MIT
