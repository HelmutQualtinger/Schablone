# Webseite „Eigenes Spraybild“: Foto hochladen, Schablonen berechnen, Simulation, STL und PNG.
#   docker build -t schablone .
#   docker run --rm -p 8000:8000 schablone      ->  http://127.0.0.1:8000
#
# Für musl gibt es auf PyPI kein OpenCV-Wheel, und Alpines py3-opencv zieht Qt, GTK, Mesa und ffmpeg mit.
# Deshalb wird OpenCV in Stufe 1 nur mit den benötigten Modulen gebaut und statisch in cv2 gelinkt.
# Alle übrigen Pakete installiert uv aus musllinux-Wheels.

ARG PYTHON=3.12
ARG ALPINE=3.22

# ---------- Stufe 1: venv mit uv, minimales OpenCV ----------
FROM python:${PYTHON}-alpine${ALPINE} AS build
COPY --from=ghcr.io/astral-sh/uv:0.13 /uv /bin/uv
RUN apk add --no-cache build-base cmake samurai linux-headers
ENV UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never UV_NO_CACHE=1 \
    UV_PYTHON=/usr/local/bin/python3 UV_PROJECT_ENVIRONMENT=/venv
WORKDIR /src
COPY pyproject.toml uv.lock ./
# Genau die Versionen aus uv.lock, ohne Gruppen (OpenCV-Wheel und reportlab braucht der Container nicht)
RUN uv sync --frozen --no-default-groups --no-install-project

ARG OPENCV=4.11.0
ADD https://github.com/opencv/opencv/archive/refs/tags/${OPENCV}.tar.gz /tmp/opencv.tar.gz
RUN mkdir /tmp/opencv && tar -xzf /tmp/opencv.tar.gz -C /tmp/opencv --strip-components=1 \
 && SITE=$(/venv/bin/python -c 'import sysconfig; print(sysconfig.get_path("platlib"))') \
 && NP=$(/venv/bin/python -c 'import numpy; print(numpy.get_include())') \
 && cmake -S /tmp/opencv -B /tmp/build -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_LIST=core,imgproc,imgcodecs,photo,python3 \
      -DBUILD_SHARED_LIBS=OFF -DBUILD_ZLIB=ON -DBUILD_PNG=ON -DBUILD_JPEG=ON \
      -DWITH_TIFF=OFF -DWITH_WEBP=OFF -DWITH_OPENJPEG=OFF -DWITH_JASPER=OFF -DWITH_OPENEXR=OFF -DWITH_IMGCODEC_HDR=OFF \
      -DWITH_IMGCODEC_SUNRASTER=OFF -DWITH_IMGCODEC_PXM=OFF -DWITH_IMGCODEC_PFM=OFF \
      -DWITH_FFMPEG=OFF -DWITH_GSTREAMER=OFF -DWITH_V4L=OFF -DWITH_GTK=OFF -DWITH_QT=OFF \
      -DWITH_OPENCL=OFF -DWITH_IPP=OFF -DWITH_ITT=OFF -DWITH_EIGEN=OFF -DWITH_LAPACK=OFF -DWITH_PROTOBUF=OFF \
      -DWITH_ADE=OFF -DWITH_OPENJPEG=OFF -DWITH_1394=OFF -DWITH_VTK=OFF \
      -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_DOCS=OFF -DBUILD_opencv_apps=OFF \
      -DBUILD_JAVA=OFF -DBUILD_opencv_python2=OFF -DOPENCV_SKIP_PYTHON_LOADER=ON \
      -DPYTHON3_EXECUTABLE=/venv/bin/python -DPYTHON3_NUMPY_INCLUDE_DIRS="$NP" \
      -DOPENCV_PYTHON3_INSTALL_PATH="$SITE" -DCMAKE_INSTALL_PREFIX=/tmp/cvinstall \
 && cmake --build /tmp/build \
 && cmake --install /tmp/build \
 && strip --strip-unneeded "$SITE"/cv2*.so \
 && /venv/bin/python -c 'import cv2; print("OpenCV", cv2.__version__)'

# Ballast entfernen: Tests, Bytecode-Caches, Typ-Stubs
RUN find /venv -depth -type d \( -name tests -o -name __pycache__ \) -exec rm -rf {} + \
 && find /venv -type f -name '*.pyi' -delete

# ---------- Stufe 2: Laufzeit ----------
FROM python:${PYTHON}-alpine${ALPINE}
RUN apk add --no-cache libstdc++ \
 && adduser -D -H -u 10001 app
COPY --from=build /venv /venv
WORKDIR /app
COPY src/separation_auto.py src/simdata.py src/druck.py src/stencil_geom.py src/motiv.py \
     src/separation.py src/separation_maria.py src/separation_carl.py ./src/
COPY web/app.py ./web/
COPY web/static ./web/static
ENV PATH=/venv/bin:$PATH PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TMPDIR=/tmp
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD wget -qO /dev/null http://127.0.0.1:8000/ || exit 1
# Ein Worker: Aufträge liegen im Prozessspeicher, die Berechnung läuft ohnehin nacheinander
CMD ["gunicorn", "--chdir", "web", "-w", "1", "--threads", "8", "-t", "300", "-b", "0.0.0.0:8000", "app:app"]
