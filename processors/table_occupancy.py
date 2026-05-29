import cv2
from processors.base_processor import BaseProcessor


class TableOccupancyProcessor(BaseProcessor):
    """
    Detecta mesas ocupadas vs libres contando personas sentadas.
    Muestra el porcentaje de ocupacion del salon.
    """

    def procesar(self, frame, resultados):
        nombres  = resultados[0].names
        cajas    = resultados[0].boxes

        personas = [
            b for b in cajas
            if nombres[int(b.cls[0])] == "person" and float(b.conf[0]) > 0.5
        ]
        sillas = [
            b for b in cajas
            if nombres[int(b.cls[0])] == "chair" and float(b.conf[0]) > 0.4
        ]

        total_sillas  = max(len(sillas), 1)
        ocupadas      = len(personas)
        porcentaje    = min(int((ocupadas / total_sillas) * 100), 100)

        self._panel(
            frame,
            f"Ocupacion: {porcentaje}%  ({ocupadas}/{total_sillas})",
            x=10, y=50,
            color_fondo=(0, 100, 200),
        )

        return frame
