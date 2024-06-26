from typing import TypedDict


class POTDict(TypedDict):
    length: float
    width: float
    height: float
    ceramic_cover_width: float
    height_with_rebar: float
    d_top: float
    d_w: float
    d_bottom_sides: float
    d_bottom_middle: float | None


HEIGHT_SHORT: float = 175.0  # height (mm) of POT beams up to a length of 6.25 m
HEIGHT_LONG: float = 230.0  # height (mmú of POT beam from length 6.50 m
WIDTH: float = 160.0  # width (mm) of POT
HEIGHT_PRECAST: float = 60.0  # height (mm) of block
CERAMIC_COVER_WIDTH: float = 15.0  # width (mm) of ceramic cover

# REINFORCEMENT
D_TOP: float = 8.0  # diameter (mm) of upper reinforcement
D_W_SHORT: float = 5.0  # diameter (mm) of shear reinforcement for short POT beams up to a length of 6.25 m
D_W_LONG: float = 6.0  # diameter (mm) of shear reinforcement for short POT beam from length 6.50 m

# POT BEAMS
POT_BEAMS: dict[str, POTDict] = {
    "POT 175": {
        "length": 1750.0,
        "width": WIDTH,
        "height": HEIGHT_PRECAST,
        "ceramic_cover_width": CERAMIC_COVER_WIDTH,
        "height_with_rebar": HEIGHT_SHORT,
        "d_top": D_TOP,
        "d_w": D_W_SHORT,
        "d_bottom_sides": 8.0,
        "d_bottom_middle": None,
    },
    "POT 200": {
        "length": 2000.0,
        "height": HEIGHT_PRECAST,
        "width": WIDTH,
        "ceramic_cover_width": CERAMIC_COVER_WIDTH,
        "height_with_rebar": HEIGHT_SHORT,
        "d_top": D_TOP,
        "d_w": D_W_SHORT,
        "d_bottom_sides": 8.0,
        "d_bottom_middle": None,
    },
    "POT 225": {
        "length": 2250.0,
        "height": HEIGHT_PRECAST,
        "width": WIDTH,
        "ceramic_cover_width": CERAMIC_COVER_WIDTH,
        "height_with_rebar": HEIGHT_SHORT,
        "d_top": D_TOP,
        "d_w": D_W_SHORT,
        "d_bottom_sides": 8.0,
        "d_bottom_middle": None,
    },
    "POT 250": {
        "length": 2500.0,
        "height": HEIGHT_PRECAST,
        "width": WIDTH,
        "ceramic_cover_width": CERAMIC_COVER_WIDTH,
        "height_with_rebar": HEIGHT_SHORT,
        "d_top": D_TOP,
        "d_w": D_W_SHORT,
        "d_bottom_sides": 8.0,
        "d_bottom_middle": None,
    },
    "POT 275": {
        "length": 2750.0,
        "height": HEIGHT_PRECAST,
        "width": WIDTH,
        "ceramic_cover_width": CERAMIC_COVER_WIDTH,
        "height_with_rebar": HEIGHT_SHORT,
        "d_top": D_TOP,
        "d_w": D_W_SHORT,
        "d_bottom_sides": 8.0,
        "d_bottom_middle": None,
    },
    "POT 300": {
        "length": 3000.0,
        "height": HEIGHT_PRECAST,
        "width": WIDTH,
        "ceramic_cover_width": CERAMIC_COVER_WIDTH,
        "height_with_rebar": HEIGHT_SHORT,
        "d_top": D_TOP,
        "d_w": D_W_SHORT,
        "d_bottom_sides": 10.0,
        "d_bottom_middle": None,
    },
    "POT 325": {
        "length": 3250.0,
        "height": HEIGHT_PRECAST,
        "width": WIDTH,
        "ceramic_cover_width": CERAMIC_COVER_WIDTH,
        "height_with_rebar": HEIGHT_SHORT,
        "d_top": D_TOP,
        "d_w": D_W_SHORT,
        "d_bottom_sides": 10.0,
        "d_bottom_middle": None,
    },
    "POT 350": {
        "length": 3500.0,
        "height": HEIGHT_PRECAST,
        "width": WIDTH,
        "ceramic_cover_width": CERAMIC_COVER_WIDTH,
        "height_with_rebar": HEIGHT_SHORT,
        "d_top": D_TOP,
        "d_w": D_W_SHORT,
        "d_bottom_sides": 10.0,
        "d_bottom_middle": None,
    },
    "POT 375": {
        "length": 3750.0,
        "height": HEIGHT_PRECAST,
        "width": WIDTH,
        "ceramic_cover_width": CERAMIC_COVER_WIDTH,
        "height_with_rebar": HEIGHT_SHORT,
        "d_top": D_TOP,
        "d_w": D_W_SHORT,
        "d_bottom_sides": 10.0,
        "d_bottom_middle": None,
    },
    "POT 400": {
        "length": 4000.0,
        "height": HEIGHT_PRECAST,
        "width": WIDTH,
        "ceramic_cover_width": CERAMIC_COVER_WIDTH,
        "height_with_rebar": HEIGHT_SHORT,
        "d_top": D_TOP,
        "d_w": D_W_SHORT,
        "d_bottom_sides": 12.0,
        "d_bottom_middle": None,
    },
    "POT 425": {
        "length": 4250.0,
        "height": HEIGHT_PRECAST,
        "width": WIDTH,
        "ceramic_cover_width": CERAMIC_COVER_WIDTH,
        "height_with_rebar": HEIGHT_SHORT,
        "d_top": D_TOP,
        "d_w": D_W_SHORT,
        "d_bottom_sides": 12.0,
        "d_bottom_middle": None,
    },
    "POT 450": {
        "length": 4500.0,
        "height": HEIGHT_PRECAST,
        "width": WIDTH,
        "ceramic_cover_width": CERAMIC_COVER_WIDTH,
        "height_with_rebar": HEIGHT_SHORT,
        "d_top": D_TOP,
        "d_w": D_W_SHORT,
        "d_bottom_sides": 12.0,
        "d_bottom_middle": 6.0,
    },
    "POT 475": {
        "length": 4750.0,
        "height": HEIGHT_PRECAST,
        "width": WIDTH,
        "ceramic_cover_width": CERAMIC_COVER_WIDTH,
        "height_with_rebar": HEIGHT_SHORT,
        "d_top": D_TOP,
        "d_w": D_W_SHORT,
        "d_bottom_sides": 12.0,
        "d_bottom_middle": 8.0,
    },
    "POT 500": {
        "length": 5000.0,
        "height": HEIGHT_PRECAST,
        "width": WIDTH,
        "ceramic_cover_width": CERAMIC_COVER_WIDTH,
        "height_with_rebar": HEIGHT_SHORT,
        "d_top": D_TOP,
        "d_w": D_W_SHORT,
        "d_bottom_sides": 12.0,
        "d_bottom_middle": 10.0,
    },
    "POT 525": {
        "length": 5250.0,
        "height": HEIGHT_PRECAST,
        "width": WIDTH,
        "ceramic_cover_width": CERAMIC_COVER_WIDTH,
        "height_with_rebar": HEIGHT_SHORT,
        "d_top": D_TOP,
        "d_w": D_W_SHORT,
        "d_bottom_sides": 12.0,
        "d_bottom_middle": 12.0,
    },
    "POT 550": {
        "length": 5500.0,
        "height": HEIGHT_PRECAST,
        "width": WIDTH,
        "ceramic_cover_width": CERAMIC_COVER_WIDTH,
        "height_with_rebar": HEIGHT_SHORT,
        "d_top": D_TOP,
        "d_w": D_W_SHORT,
        "d_bottom_sides": 12.0,
        "d_bottom_middle": 12.0,
    },
    "POT 575": {
        "length": 5750.0,
        "height": HEIGHT_PRECAST,
        "width": WIDTH,
        "ceramic_cover_width": CERAMIC_COVER_WIDTH,
        "height_with_rebar": HEIGHT_SHORT,
        "d_top": D_TOP,
        "d_w": D_W_SHORT,
        "d_bottom_sides": 12.0,
        "d_bottom_middle": 12.0,
    },
    "POT 600": {
        "length": 6000.0,
        "height": HEIGHT_PRECAST,
        "width": WIDTH,
        "ceramic_cover_width": CERAMIC_COVER_WIDTH,
        "height_with_rebar": HEIGHT_SHORT,
        "d_top": D_TOP,
        "d_w": D_W_SHORT,
        "d_bottom_sides": 12.0,
        "d_bottom_middle": 14.0,
    },
    "POT 625": {
        "length": 6250.0,
        "height": HEIGHT_PRECAST,
        "width": WIDTH,
        "ceramic_cover_width": CERAMIC_COVER_WIDTH,
        "height_with_rebar": HEIGHT_SHORT,
        "d_top": D_TOP,
        "d_w": D_W_SHORT,
        "d_bottom_sides": 12.0,
        "d_bottom_middle": 14.0,
    },
    "POT 650": {
        "length": 6500.0,
        "height": HEIGHT_PRECAST,
        "width": WIDTH,
        "ceramic_cover_width": CERAMIC_COVER_WIDTH,
        "height_with_rebar": HEIGHT_LONG,
        "d_top": D_TOP,
        "d_w": D_W_LONG,
        "d_bottom_sides": 12.0,
        "d_bottom_middle": 14.0,
    },
    "POT 675": {
        "length": 6750.0,
        "height": HEIGHT_PRECAST,
        "width": WIDTH,
        "ceramic_cover_width": CERAMIC_COVER_WIDTH,
        "height_with_rebar": HEIGHT_LONG,
        "d_top": D_TOP,
        "d_w": D_W_LONG,
        "d_bottom_sides": 12.0,
        "d_bottom_middle": 16.0,
    },
    "POT 700": {
        "length": 7000.0,
        "height": HEIGHT_PRECAST,
        "width": WIDTH,
        "ceramic_cover_width": CERAMIC_COVER_WIDTH,
        "height_with_rebar": HEIGHT_LONG,
        "d_top": D_TOP,
        "d_w": D_W_LONG,
        "d_bottom_sides": 12.0,
        "d_bottom_middle": 18.0,
    },
    "POT 725": {
        "length": 7250.0,
        "height": HEIGHT_PRECAST,
        "width": WIDTH,
        "ceramic_cover_width": CERAMIC_COVER_WIDTH,
        "height_with_rebar": HEIGHT_LONG,
        "d_top": D_TOP,
        "d_w": D_W_LONG,
        "d_bottom_sides": 12.0,
        "d_bottom_middle": 18.0,
    },
    "POT 750": {
        "length": 7500.0,
        "height": HEIGHT_PRECAST,
        "width": WIDTH,
        "ceramic_cover_width": CERAMIC_COVER_WIDTH,
        "height_with_rebar": HEIGHT_LONG,
        "d_top": D_TOP,
        "d_w": D_W_LONG,
        "d_bottom_sides": 12.0,
        "d_bottom_middle": 18.0,
    },
    "POT 775": {
        "length": 7750.0,
        "height": HEIGHT_PRECAST,
        "width": WIDTH,
        "ceramic_cover_width": CERAMIC_COVER_WIDTH,
        "height_with_rebar": HEIGHT_LONG,
        "d_top": D_TOP,
        "d_w": D_W_LONG,
        "d_bottom_sides": 12.0,
        "d_bottom_middle": 20.0,
    },
    "POT 8000": {
        "length": 8000.0,
        "height": HEIGHT_PRECAST,
        "width": WIDTH,
        "ceramic_cover_width": CERAMIC_COVER_WIDTH,
        "height_with_rebar": HEIGHT_LONG,
        "d_top": D_TOP,
        "d_w": D_W_LONG,
        "d_bottom_sides": 12.0,
        "d_bottom_middle": 20.0,
    },
    "POT 825": {
        "length": 8250.0,
        "height": HEIGHT_PRECAST,
        "width": WIDTH,
        "ceramic_cover_width": CERAMIC_COVER_WIDTH,
        "height_with_rebar": HEIGHT_LONG,
        "d_top": D_TOP,
        "d_w": D_W_LONG,
        "d_bottom_sides": 12.0,
        "d_bottom_middle": 20.0,
    },
}
