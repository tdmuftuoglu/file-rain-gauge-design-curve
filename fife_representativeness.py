# =============================================================================
# How many rain gauges are needed to represent areal rainfall?
# Point-to-areal representativeness from two dense rain-gauge networks:
#
#   (A) FIFE / Konza Prairie, Kansas (sub-humid; 1982-1989; ~22 km2)
#       Daily Rainfall Data (FIFE), ORNL DAAC, doi:10.3334/ORNLDAAC/29
#       30 Minute Rainfall Data (FIFE), ORNL DAAC, doi:10.3334/ORNLDAAC/2
#   (B) Walnut Gulch Experimental Watershed, Arizona (semi-arid, monsoon
#       thunderstorms; digital gauges 2000-present; ~148 km2)
#       USDA-ARS Southwest Watershed Research Center, Data Access Project:
#       https://www.tucson.ars.ag.gov/dap/
#         - Precipitation > Digital > Aggregate data: watershed 63, all gauges,
#           time step Daily, units mm, no statistics; saved as dap_aggregate.xls
#         - GIS Data Layers > "Instrumentation" (instrumentation.zip) and
#           "Boundary" (boundary.zip)
#
# What the script does
#   1. Reads and quality-controls both networks (duplicates, outages, a spatial
#      neighbour check, collocated and cross-network gauge comparisons).
#   2. Builds design curves: the relative error of a k-gauge areal mean against
#      (i) the mean of all gauges (nested), (ii) the mean of the remaining gauges
#      (disjoint) and (iii) the true areal mean for randomly placed gauges.
#   3. Derives distribution-free (conformal) 90% tolerance radii and checks their
#      coverage across periods, seasons, storm sizes and networks.
#   4. Examines variance-mean scaling, a gradient-boosting model of spatial
#      variability (cross-validated, with out-of-fold SHAP values), inter-gauge
#      correlation versus error, Thiessen weighting and sub-daily accumulations.
#
# How to run (Google Colab): paste the whole script into ONE cell and run it.
# A "Choose Files" button appears; select all input files at once:
#   FIFE daily archive (required), FIFE 30-minute archive (optional),
#   dap_aggregate.xls + instrumentation.zip + boundary.zip (optional, network B).
# Any of .rar / .zip / .tar / .tar.gz is accepted. All figures, tables and
# numbers are written to fife_full_outputs/ and downloaded as a zip file.
# Random seeds are fixed, so repeated runs give identical numbers.
# =============================================================================
import os, re, sys, glob, subprocess, itertools, warnings, shutil, math
warnings.filterwarnings('ignore')

WORKDIR = 'fife_work'
OUTDIR = 'fife_full_outputs'
IN_COLAB = 'google.colab' in sys.modules
for d in (WORKDIR, OUTDIR):                 # start from empty folders on every run
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d, exist_ok=True)

MISSING = -9.9          # missing-value code of the daily archive
WET_MM = 0.5            # a day (or interval) is wet when the areal mean exceeds this
RAIN_MM = 0.2           # a gauge reports rain when its total exceeds this
DENSE_MIN = 20          # dense analysis set: wet days with >= 20 reporting sites
ROUTINE_MIN = 7         # routine analysis set: wet days with 7-19 reporting sites
ALPHA = 0.10            # 1 - nominal coverage of the tolerance radius
NB_RADIUS = 1500.0      # neighbourhood radius (m) for spatial consistency checks
SEED = 2024


def _pip(pkg):
    r = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', pkg], check=False)
    if r.returncode:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                        '--break-system-packages', pkg], check=False)


for _mod, _pkg in [('libarchive', 'libarchive-c'), ('pyproj', 'pyproj'),
                   ('shapely', 'shapely'), ('lightgbm', 'lightgbm'), ('shap', 'shap'),
                   ('shapefile', 'pyshp'), ('lxml', 'lxml')]:
    try:
        __import__(_mod)
    except ImportError:
        _pip(_pkg)

import libarchive, pyproj
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats, optimize
from scipy.spatial import Voronoi
from shapely.geometry import Polygon, MultiPoint
import lightgbm as lgb
import shap

plt.rcParams.update({'figure.dpi': 130, 'savefig.dpi': 300, 'font.size': 9,
                     'axes.grid': True, 'grid.alpha': .3, 'savefig.bbox': 'tight'})
BLUE, RED, GREEN, GREY, ORANGE, PURPLE = '#1b6ca8', '#d1495b', '#2a9d8f', '#888888', '#e9a03b', '#6a4c93'

SUMMARY = []


def note(s=''):
    print(s); SUMMARY.append(s)


# =============================================================================
# 1. INPUT FILES
# =============================================================================
ARCH_EXT = ('.rar', '.zip', '.tar', '.gz', '.tgz', '.7z', '.bz2', '.xz')


def _local_inputs():
    hits = []
    for p in sorted(glob.glob('*')):
        if os.path.isfile(p) and (p.lower().endswith(ARCH_EXT) or
                                  re.search(r'\.(rdy|r30|txt|dat|xls|html?)$', p.lower())):
            hits.append(p)
    return hits


if IN_COLAB:
    # the input files are always chosen interactively in Colab
    from google.colab import files
    print('>> Select all input files at once: the FIFE daily archive, the FIFE 30-minute archive, '
          'dap_aggregate.xls, instrumentation.zip and boundary.zip')
    up = files.upload()
    INPUTS = list(up.keys())
else:
    INPUTS = _local_inputs()
    if not INPUTS:
        raise FileNotFoundError('Place the input files next to this script and re-run.')
print('Input files:', INPUTS)


def _extract(path, dst, depth=0):
    """Extract every file entry of an archive (recursively for nested archives)."""
    n = 0
    try:
        with libarchive.file_reader(path) as arch:
            for e in arch:
                if e.isdir or e.size == 0:
                    continue
                name = os.path.basename(e.pathname)
                out = os.path.join(dst, '%d_%s' % (depth, name))
                k = 1
                while os.path.exists(out):
                    out = os.path.join(dst, '%d_%d_%s' % (depth, k, name)); k += 1
                with open(out, 'wb') as fh:
                    for blk in e.get_blocks():
                        fh.write(blk)
                n += 1
                if out.lower().endswith(ARCH_EXT) and depth < 3:
                    n += _extract(out, dst, depth + 1)
    except Exception as exc:
        print('  could not open %s as an archive (%s); treated as a plain file' % (path, exc))
        shutil.copy(path, os.path.join(dst, os.path.basename(path)))
        n += 1
    return n


for p in INPUTS:
    if p.lower().endswith(ARCH_EXT):
        print('Extracted %d entries from %s' % (_extract(p, WORKDIR), p))
    else:
        shutil.copy(p, os.path.join(WORKDIR, os.path.basename(p)))


# =============================================================================
# 2. PARSE GRANULES (daily and 30-minute tables are recognised by their columns)
# =============================================================================
DAILY_COLS = ['SITEGRID_ID', 'STATION_ID', 'OBS_DATE', 'OBS_TYPE', 'TOTAL_RAINFALL',
              'COMMENTS', 'FIFE_DATA_CRTFCN_CODE', 'LAST_REVISION_DATE']
MIN30_COLS = ['SITEGRID_ID', 'STATION_ID', 'OBS_DATE', 'OBS_TIME', 'LOCAL_OBS_DATE',
              'LOCAL_OBS_TIME', 'RAINFALL', 'FIFE_DATA_CRTFCN_CODE', 'LAST_REVISION_DATE']
daily_rows, min30_rows = [], []
for f in sorted(glob.glob(os.path.join(WORKDIR, '*'))):
    if f.lower().endswith(ARCH_EXT):
        continue
    try:
        txt = open(f, encoding='latin-1').read().replace('\x1a', '')
    except Exception:
        continue
    lines = [l.strip() for l in re.split(r'[\r\n]+', txt) if l.strip()]
    hdr = [i for i, l in enumerate(lines) if l.upper().startswith('SITEGRID_ID')]
    if not hdr:
        continue
    cols = [c.strip().strip("'").upper() for c in lines[hdr[0]].split(',')]
    if 'TOTAL_RAINFALL' in cols or len(cols) == 8:
        target, names = daily_rows, (cols if len(cols) == 8 else DAILY_COLS)
    elif 'LOCAL_OBS_TIME' in cols or len(cols) == 9:
        target, names = min30_rows, (cols if len(cols) == 9 else MIN30_COLS)
    else:
        continue
    for l in lines[hdr[0] + 1:]:
        p = [x.strip().strip("'").strip() for x in l.split(',')]
        if len(p) >= len(names):
            target.append(dict(zip(names, p[:len(names)])))

if not daily_rows:
    raise RuntimeError('No daily-rainfall granules were found in the selected files.')
HAVE_30 = len(min30_rows) > 0
print('Parsed %d daily records%s' % (len(daily_rows),
      (' and %d 30-minute records' % len(min30_rows)) if HAVE_30 else ' (no 30-minute data supplied)'))


def _fife_date(s):
    d = pd.to_datetime(s, format='%d-%b-%y', errors='coerce')
    return d.where(d.dt.year < 2000, d - pd.DateOffset(years=100))


df = pd.DataFrame(daily_rows).rename(columns={'SITEGRID_ID': 'sitegrid', 'STATION_ID': 'station',
                                              'OBS_DATE': 'date', 'OBS_TYPE': 'obstype',
                                              'TOTAL_RAINFALL': 'rain'})
df['rain'] = pd.to_numeric(df['rain'], errors='coerce')
df['station'] = pd.to_numeric(df['station'], errors='coerce').astype(int)
df['date'] = _fife_date(df['date'])
assert df['date'].notna().all(), 'unparsed dates present'
df['network'] = np.where(df.sitegrid.str.endswith('RGP'), 'Princeton', 'LTER')
df['raw_missing'] = df.rain <= MISSING + 1e-6
df.loc[df.raw_missing, 'rain'] = np.nan

# =============================================================================
# 3. GAUGE INVENTORY AND COORDINATES
#    Station coordinates (UTM zone 14N) as tabulated in the ORNL DAAC user guide
#    of the daily data set; checked below against the FIS site-grid codes.
# =============================================================================
DOC_UTM = {100: (4330786, 706889), 225: (4326820, 709650), 226: (4327950, 708530),
           227: (4329040, 709000), 228: (4328970, 710780), 229: (4327010, 706830),
           230: (4328950, 708080), 231: (4330650, 706780), 235: (4330388, 706824),
           301: (4331360, 708524), 302: (4330561, 708640), 303: (4330732, 709244),
           304: (4330841, 710152), 305: (4330256, 708498), 306: (4330366, 709866),
           307: (4330427, 710750), 308: (4329376, 708140), 309: (4329924, 709063),
           310: (4329811, 709640), 311: (4329933, 710158), 312: (4329723, 711329),
           313: (4329146, 708577), 314: (4329309, 709212), 315: (4329256, 709754),
           316: (4329457, 710188), 317: (4329500, 710876), 318: (4329113, 711634),
           319: (4328884, 708195), 320: (4328921, 709272), 321: (4328671, 709549),
           322: (4328909, 709961), 323: (4328823, 710821), 324: (4328488, 707878),
           325: (4328335, 708701), 326: (4327854, 709293), 327: (4328244, 710726),
           328: (4328128, 711091), 329: (4327746, 708927), 330: (4327805, 709276),
           332: (4329354, 708156), 334: (4328327, 709498)}
GRID_N0, GRID_E0, CELL = 4_334_000.0, 705_000.0, 200.0
tr = pyproj.Transformer.from_crs('EPSG:32614', 'EPSG:4326', always_xy=True)

inv = (df.groupby(['station', 'sitegrid', 'network'], as_index=False)
         .agg(n_records=('rain', 'size'), n_valid=('rain', 'count'),
              first=('date', 'min'), last=('date', 'max')))
inv['grid_code'] = inv.sitegrid.str[:4]
inv['georef'] = inv.grid_code.str.fullmatch(r'\d{4}') & inv.station.isin(DOC_UTM.keys())
inv['utm_n'] = inv.station.map(lambda s: DOC_UTM.get(s, (np.nan, np.nan))[0])
inv['utm_e'] = inv.station.map(lambda s: DOC_UTM.get(s, (np.nan, np.nan))[1])
gn = GRID_N0 - pd.to_numeric(inv.grid_code.where(inv.georef).str[:2]) * CELL
ge = GRID_E0 + pd.to_numeric(inv.grid_code.where(inv.georef).str[2:]) * CELL
inv['grid_offset_m'] = np.hypot(inv.utm_n - (gn - CELL / 2), inv.utm_e - (ge + CELL / 2))
inv['lon'], inv['lat'] = tr.transform(inv.utm_e.values, inv.utm_n.values)

note('=== Gauge inventory ===')
note('station records in archive: %d (LTER %d, Princeton %d) at %d site-grid locations'
     % (len(inv), (inv.network == 'LTER').sum(), (inv.network == 'Princeton').sum(),
        inv.sitegrid.nunique()))
note('  without grid coordinates: %s' % ', '.join(inv.loc[~inv.georef, 'sitegrid']))
note('  without any valid record: %s' % (', '.join(inv.loc[inv.n_valid == 0, 'sitegrid']) or 'none'))
coll = inv[inv.georef].groupby('sitegrid').station.apply(list)
coll = coll[coll.map(len) > 1]
note('  collocated gauges (two instruments in one grid cell): %s'
     % '; '.join('%s: stations %s' % (k, v) for k, v in coll.items()))
note('  documented coordinates vs grid-cell centres: max offset %.0f m'
     % inv.grid_offset_m.max())

# =============================================================================
# 4. RECORD-LEVEL QUALITY CONTROL
# =============================================================================
qc_log = []
neg = df.rain < 0
big = df.rain > 200
df.loc[neg | big, 'rain'] = np.nan
qc_log.append(('negative or >200 mm values set to missing', int((neg | big).sum())))

dup = df.groupby(['station', 'date']).rain.agg(['size', 'count', 'min', 'max'])
conf = dup[(dup['count'] > 1) & ((dup['max'] - dup['min']) > 0.05)]
same = dup[(dup['size'] > 1) & ~dup.index.isin(conf.index)]
qc_log.append(('duplicate station-days with identical values (kept once)', len(same)))
qc_log.append(('duplicate station-days with conflicting values (set to missing)', len(conf)))
conf_tab = conf.reset_index().merge(inv[['station', 'sitegrid']], on='station')
st = df.groupby(['station', 'date'], as_index=False).rain.mean()
st = st.merge(conf.reset_index()[['station', 'date']].assign(conflict=True),
              on=['station', 'date'], how='left')
st.loc[st.conflict.fillna(False).astype(bool), 'rain'] = np.nan
st = st.drop(columns='conflict')
S = st.pivot(index='date', columns='station', values='rain').sort_index()   # station x day

# collocated-instrument repeatability (before the two are merged into one site)
coll_rows = []
for sg, ids in coll.items():
    a, b = ids[0], ids[1]
    x = S[[a, b]].dropna(); x = x[(x[a] > RAIN_MM) | (x[b] > RAIN_MM)]
    if len(x) >= 5:
        d = float(np.hypot(*(np.subtract(DOC_UTM[a], DOC_UTM[b]))))
        coll_rows.append((sg, a, b, d, len(x), x[a].corr(x[b]),
                          float((x[a] - x[b]).abs().median()),
                          float(np.sqrt(((x[a] - x[b]) ** 2).mean())),
                          float(np.sqrt(((x[a] - x[b]) ** 2).mean()) / x.values.mean()),
                          float(x[b].sum() / x[a].sum())))
COLL = pd.DataFrame(coll_rows, columns=['sitegrid', 'station_a', 'station_b', 'separation_m',
                                        'n_rain_days', 'pearson_r', 'median_abs_diff_mm',
                                        'rmsd_mm', 'nrmsd', 'ratio_b_to_a'])

# site-level matrix: georeferenced stations, collocated instruments averaged
geo_st = inv[inv.georef & (inv.n_valid > 0)]
site_of = dict(zip(geo_st.station, geo_st.sitegrid))
P_all = S[[c for c in S.columns if c in site_of]].T.groupby(site_of).mean().T.sort_index()
P_all = P_all[sorted(P_all.columns)]
sites = (geo_st.groupby('sitegrid', as_index=False)
               .agg(network=('network', 'first'), utm_n=('utm_n', 'mean'), utm_e=('utm_e', 'mean'),
                    n_instruments=('station', 'size')))
sites['lon'], sites['lat'] = tr.transform(sites.utm_e.values, sites.utm_n.values)
sites = sites.set_index('sitegrid').reindex(list(P_all.columns)).rename_axis('sitegrid').reset_index()
XY = sites[['utm_e', 'utm_n']].values
coord = {r.sitegrid: (r.utm_e, r.utm_n) for r in sites.itertuples()}
net_of = dict(zip(sites.sitegrid, sites.network))
DIST = np.sqrt(((XY[:, None, :] - XY[None, :, :]) ** 2).sum(-1))

# ---- spatial consistency check (neighbours within NB_RADIUS) ----
def neighbour_flags(A0, DIST, radius, min_nb=3):
    """Spatial consistency check of a day x gauge matrix.
    low : value <= 25% of the neighbour median while every neighbour recorded >= 2 mm
          (consistent with a partial or lost record);
    high: value >= 4x the neighbour median and >= 10 mm above every neighbour."""
    flag_low = np.zeros_like(A0, dtype=bool); flag_high = np.zeros_like(A0, dtype=bool)
    nbmask = (DIST <= radius) & (DIST > 0)
    for j in range(A0.shape[1]):
        nb = np.where(nbmask[j])[0]
        if len(nb) < min_nb:
            continue
        V = A0[:, nb]
        cnt = np.sum(~np.isnan(V), 1)
        with np.errstate(all='ignore'):
            med = np.nanmedian(V, 1); mn = np.nanmin(V, 1); mx = np.nanmax(V, 1)
        x = A0[:, j]; ok = (cnt >= min_nb) & ~np.isnan(x)
        flag_low[:, j] = ok & (mn >= 2.0) & (x <= 0.25 * med)
        flag_high[:, j] = ok & (x >= 4 * med) & (x - mx >= 10.0)
    return flag_low, flag_high, nbmask


A0 = P_all.values.copy()
flag_low, flag_high, nbmask = neighbour_flags(A0, DIST, NB_RADIUS)
qc_log.append(('site-days flagged low (<=25% of neighbour median while all neighbours >=2 mm)',
               int(flag_low.sum())))
qc_log.append(('site-days flagged high (>=4x neighbour median and >=10 mm above every neighbour)',
               int(flag_high.sum())))
FLAGS = []
for (i, j) in zip(*np.where(flag_low | flag_high)):
    FLAGS.append((P_all.index[i].date(), P_all.columns[j], net_of[P_all.columns[j]],
                  'low' if flag_low[i, j] else 'high', A0[i, j]))
FLAGS = pd.DataFrame(FLAGS, columns=['date', 'site', 'network', 'flag', 'value_mm'])

P_noscreen = P_all.copy()                                 # sensitivity variant 1
P_all = P_all.mask(pd.DataFrame(flag_low, index=P_all.index, columns=P_all.columns))   # primary
P_strict = P_all.mask(pd.DataFrame(flag_high, index=P_all.index, columns=P_all.columns))  # variant 2

# ---- site-to-neighbourhood ratio (double-mass type consistency) ----
ratio_rows = []
for j, sg in enumerate(P_all.columns):
    nb = np.where(nbmask[j])[0]
    if len(nb) < 2:
        continue
    x = P_all.iloc[:, j]; nbm = P_all.iloc[:, nb].mean(1); cnt = P_all.iloc[:, nb].notna().sum(1)
    ok = x.notna() & (cnt >= 2) & (nbm > WET_MM)
    if ok.sum() >= 10:
        ratio_rows.append((sg, net_of[sg], int(ok.sum()), float(x[ok].sum() / nbm[ok].sum())))
RATIO = pd.DataFrame(ratio_rows, columns=['site', 'network', 'n_days', 'site_to_neighbour_ratio'])
RATIO['outside_0.8_1.25'] = ~RATIO.site_to_neighbour_ratio.between(0.8, 1.25)

note('\n=== Quality control ===')
for k_, v_ in qc_log:
    note('  %-88s %d' % (k_, v_))
if len(conf_tab):
    note('  conflicting duplicates on: %s' % ', '.join(sorted({str(d.date()) for d in conf_tab.date})))
if len(FLAGS):
    note('  flagged site-days:\n' + FLAGS.to_string(index=False))
note('  site-to-neighbourhood totals ratio: median %.2f, range %.2f-%.2f, %d of %d sites outside 0.80-1.25'
     % (RATIO.site_to_neighbour_ratio.median(), RATIO.site_to_neighbour_ratio.min(),
        RATIO.site_to_neighbour_ratio.max(), int(RATIO['outside_0.8_1.25'].sum()), len(RATIO)))
note('  collocated instruments:\n' + COLL.round(3).to_string(index=False))

# =============================================================================
# 5. NETWORK GEOMETRY, DAILY COMPOSITION AND ANALYSIS SETS
# =============================================================================
span_n = (sites.utm_n.max() - sites.utm_n.min()) / 1000
span_e = (sites.utm_e.max() - sites.utm_e.min()) / 1000
HULL = MultiPoint([tuple(p) for p in XY]).convex_hull
D_ = DIST.copy(); np.fill_diagonal(D_, np.inf)
note('\n=== Network geometry (georeferenced sites with data) ===')
note('sites: %d (LTER %d, Princeton %d); instruments: %d'
     % (len(sites), (sites.network == 'LTER').sum(), (sites.network == 'Princeton').sum(),
        int(sites.n_instruments.sum())))
note('bounding box %.1f km (N-S) x %.1f km (E-W) = %.1f km2; convex hull %.1f km2'
     % (span_n, span_e, span_n * span_e, HULL.area / 1e6))
note('lat %.3f..%.3f N, lon %.3f..%.3f E'
     % (sites.lat.min(), sites.lat.max(), sites.lon.min(), sites.lon.max()))
note('nearest-neighbour distance: min %.0f m | median %.0f m | max %.0f m'
     % (D_.min(1).min(), np.median(D_.min(1)), D_.min(1).max()))
note('maximum separation between sites: %.0f m' % DIST.max())

LT = [c for c in P_all.columns if net_of[c] == 'LTER']
PR = [c for c in P_all.columns if net_of[c] == 'Princeton']


def _hull_area(cols):
    pts = [coord[c] for c in cols]
    return MultiPoint(pts).convex_hull.area / 1e6 if len(pts) >= 3 else np.nan


daily = pd.DataFrame({'n_sites': P_all.notna().sum(1),
                      'n_lter': P_all[LT].notna().sum(1),
                      'n_princeton': P_all[PR].notna().sum(1),
                      'areal_mean': P_all.mean(1), 'areal_sd': P_all.std(1),
                      'areal_min': P_all.min(1), 'areal_max': P_all.max(1),
                      'wet_fraction': (P_all > RAIN_MM).sum(1) / P_all.notna().sum(1).replace(0, np.nan)})
daily['cv'] = daily.areal_sd / daily.areal_mean.replace(0, np.nan)
daily['range_mm'] = daily.areal_max - daily.areal_min
daily['wet'] = daily.areal_mean > WET_MM
daily['set'] = np.where(daily.wet & (daily.n_sites >= DENSE_MIN), 'dense',
                        np.where(daily.wet & (daily.n_sites >= ROUTINE_MIN), 'routine', ''))
daily['hull_km2'] = [(_hull_area(P_all.columns[P_all.loc[d].notna().values])
                      if s else np.nan) for d, s in zip(daily.index, daily['set'])]
daily['km2_per_site'] = daily.hull_km2 / daily.n_sites
daily['year'] = daily.index.year
Dd = daily[daily['set'] == 'dense']; Rd = daily[daily['set'] == 'routine']


def _rng(s):
    return '%d (%d-%d)' % (s.median(), s.min(), s.max()) if len(s) else '-'


comp = []
for name, sub in [('dense', Dd), ('routine', Rd)]:
    for yr, g in list(sub.groupby('year')) + [('all', sub)]:
        comp.append((name, yr, len(g), _rng(g.n_sites), _rng(g.n_lter), _rng(g.n_princeton),
                     '%.1f' % g.hull_km2.median() if len(g) else '-',
                     '%.2f' % g.km2_per_site.median() if len(g) else '-',
                     '%s-%s' % (g.index.min().strftime('%Y-%m-%d'), g.index.max().strftime('%Y-%m-%d'))
                     if len(g) else '-'))
COMP = pd.DataFrame(comp, columns=['set', 'year', 'wet_days', 'sites_per_day', 'lter_per_day',
                                   'princeton_per_day', 'median_hull_km2', 'median_km2_per_site', 'period'])
note('\n=== Daily network composition (wet days) ===')
note(COMP.to_string(index=False))
note('maximum number of sites reporting on one day: %d (on %d days)'
     % (daily.n_sites.max(), int((daily.n_sites == daily.n_sites.max()).sum())))
for name, sub in [('dense', Dd), ('routine', Rd)]:
    part = sub[(sub.wet_fraction > 0) & (sub.wet_fraction < 1)]
    note('%s set: spatial CV median %.2f (IQR %.2f-%.2f); max-min spread median %.1f mm, p90 %.1f, max %.1f; '
         'partially wetted days %d/%d (%.0f%%)'
         % (name, sub.cv.median(), sub.cv.quantile(.25), sub.cv.quantile(.75), sub.range_mm.median(),
            sub.range_mm.quantile(.9), sub.range_mm.max(), len(part), len(sub), 100 * len(part) / max(len(sub), 1)))

# =============================================================================
# 6. LTER vs PRINCETON INSTRUMENTS (overlap period)
# =============================================================================
ov = P_all.loc[(P_all[PR].notna().sum(1) >= 1) & (P_all[LT].notna().sum(1) >= 1)]
ovp = (ov.index.min(), ov.index.max())


def _pair_stats(a, b, A):
    x = A[[a, b]].dropna(); x = x[(x[a] > RAIN_MM) | (x[b] > RAIN_MM)]
    if len(x) < 8:
        return None
    lagr = {}
    for lag in (-1, 0, 1):
        y = pd.concat([A[a], A[b].shift(lag, freq='D')], axis=1).dropna()
        y = y[(y.iloc[:, 0] > RAIN_MM) | (y.iloc[:, 1] > RAIN_MM)]
        lagr[lag] = y.iloc[:, 0].corr(y.iloc[:, 1]) if len(y) >= 8 else np.nan
    w = stats.wilcoxon(x[b] - x[a]).pvalue if (x[b] - x[a]).abs().sum() > 0 else np.nan
    return dict(n=len(x), ratio=float(x[b].sum() / x[a].sum()),
                median_diff=float((x[b] - x[a]).median()), wilcoxon_p=float(w),
                r_lag0=lagr[0], r_lagm1=lagr[-1], r_lagp1=lagr[1],
                wetdry=float(((x[a] > RAIN_MM) == (x[b] > RAIN_MM)).mean()))


pair_rows = []
Aov = P_all.loc[ovp[0]:ovp[1]]
cols_ = list(P_all.columns)
for i, j in itertools.combinations(range(len(cols_)), 2):
    d = DIST[i, j]
    if d > 500:
        continue
    a, b = cols_[i], cols_[j]
    ta, tb = net_of[a], net_of[b]
    if ta == 'Princeton' and tb == 'LTER':          # LTER gauge listed first in mixed pairs
        a, b = b, a; ta, tb = tb, ta
    kind = 'LTER-Princeton' if ta != tb else ('Princeton-Princeton' if ta == 'Princeton' else 'LTER-LTER')
    if kind == 'LTER-LTER':
        continue
    s_ = _pair_stats(a, b, Aov)
    if s_:
        pair_rows.append(dict(site_a=a, site_b=b, kind=kind, dist_m=d, **s_))
NETCMP = pd.DataFrame(pair_rows)

lm, pm = Aov[LT].mean(1), Aov[PR].mean(1)
okn = (Aov[LT].notna().sum(1) >= 5) & (Aov[PR].notna().sum(1) >= 10)
net_lag = {}
for lag in (-1, 0, 1):
    y = pd.concat([lm[okn], pm[okn].shift(lag, freq='D')], axis=1).dropna()
    y = y[(y.iloc[:, 0] > WET_MM) | (y.iloc[:, 1] > WET_MM)]
    net_lag[lag] = (y.iloc[:, 0].corr(y.iloc[:, 1]), len(y))
yy = pd.concat([lm, pm], axis=1)[okn].dropna(); yy = yy[(yy.iloc[:, 0] > WET_MM) | (yy.iloc[:, 1] > WET_MM)]
note('\n=== LTER (weighing) vs Princeton (tipping-bucket) gauges, overlap %s to %s ==='
     % (ovp[0].date(), ovp[1].date()))
if len(NETCMP):
    for kind, g in NETCMP.groupby('kind'):
        note('  %s pairs <=500 m: n=%d | separation %.0f-%.0f m | totals ratio median %.3f (range %.3f-%.3f) | '
             'median daily diff %.2f mm | r lag0 %.3f vs lag-1 %.3f / lag+1 %.3f (medians) | wet/dry agreement %.2f | '
             'Wilcoxon p<0.05 in %d pairs'
             % (kind, len(g), g.dist_m.min(), g.dist_m.max(), g.ratio.median(), g.ratio.min(), g.ratio.max(),
                g.median_diff.median(), g.r_lag0.median(), g.r_lagm1.median(), g.r_lagp1.median(),
                g.wetdry.median(), int((g.wilcoxon_p < 0.05).sum())))
    note(NETCMP.round(3).to_string(index=False))
note('  network means (>=5 LTER and >=10 Princeton sites, n=%d rainy days): Princeton/LTER totals ratio %.3f; '
     'r lag0 %.3f, lag-1 %.3f, lag+1 %.3f'
     % (len(yy), yy.iloc[:, 1].sum() / yy.iloc[:, 0].sum(), net_lag[0][0], net_lag[-1][0], net_lag[1][0]))

# =============================================================================
# 7. SUBSET-SAMPLING ENGINE
# =============================================================================
rng0 = np.random.default_rng(SEED)


def draws_km(v, k, m, R, rng):
    """R random disjoint draws: mean of k sites and mean of the next m sites."""
    n = len(v)
    perm = np.argsort(rng.random((R, n)), 1)
    cs = np.cumsum(v[perm], 1)
    est = cs[:, k - 1] / k
    ref = (cs[:, k + m - 1] - cs[:, k - 1]) / m if m > 0 else None
    return est, ref


def day_values(P, idx):
    out = []
    for d in idx:
        v = P.loc[d].dropna()
        out.append((d, v.index.values, v.values.astype(float)))
    return out


def design_curves(P, idx, ks, R=200, seed=1):
    """Nested (reference = all sites) and disjoint (reference = complementary sites) curves."""
    rng = np.random.default_rng(seed); dv = day_values(P, idx); out = {}
    for k in ks:
        rec = []
        for d, names, v in dv:
            n = len(v)
            if n < 2 * k:
                continue
            Rall = v.mean()
            est, ref = draws_km(v, k, n - k, R, rng)
            e_nest = (est - Rall) / Rall
            e_disj = (est - ref) / Rall
            rec.append(dict(date=d, n=n, mse_nest=np.mean(e_nest ** 2), mse_disj=np.mean(e_disj ** 2),
                            w20=np.mean(np.abs(e_disj) <= .2), w10=np.mean(np.abs(e_disj) <= .1),
                            det=np.mean((est > WET_MM) == (ref > WET_MM)),
                            mae_mm=np.mean(np.abs(est - ref))))
        out[k] = pd.DataFrame(rec)
    return out


def _boot_ci(x, B=2000, seed=0):
    """Day-cluster bootstrap of sqrt(mean(x)) (x = per-day mean squared error): percentile and BCa."""
    x = np.asarray(x, float); rng = np.random.default_rng(seed)
    bt = np.array([np.sqrt(x[rng.integers(0, len(x), len(x))].mean()) for _ in range(B)]) * 100
    pct = np.percentile(bt, [2.5, 97.5])
    try:
        kw = dict(vectorized=False, method='BCa', n_resamples=B, confidence_level=.95)
        f = lambda a: np.sqrt(np.mean(a)) * 100
        try:
            res = stats.bootstrap((x,), f, rng=np.random.default_rng(seed), **kw)
        except TypeError:
            res = stats.bootstrap((x,), f, random_state=seed, **kw)
        bca = (res.confidence_interval.low, res.confidence_interval.high)
    except Exception:
        bca = (np.nan, np.nan)
    return pct, bca, stats.skew(x)


def truth_curve(P, idx, B=2000, seed=3):
    """Error of a k-gauge mean against the TRUE areal mean when the k gauges are placed at random
    within the network footprint.  For a day with site values v (n sites, sample variance S2, mean R)
    a random k-subset mean has expected squared relative error S2 / (k R2) against the field mean
    (the n sites being treated as a sample of point locations in the area).  Averaged over days:
        MSE_true(k)   = mean(CV2) / k
        MSE_nested(k) = mean(CV2 (1/k - 1/n))      (reference = all n sites, finite-population effect)
        MSE_disj(k)   = mean(CV2 (1/k + 1/(n-k)))  (reference = the other n-k sites)
    with CV2 = S2 / R2.  Confidence limits: day-cluster bootstrap of mean(CV2)."""
    dv = day_values(P, idx)
    cv2 = np.array([v.var(ddof=1) / v.mean() ** 2 for _, _, v in dv])
    n = np.array([len(v) for _, _, v in dv])
    rng = np.random.default_rng(seed)
    boot = np.array([cv2[rng.integers(0, len(cv2), len(cv2))].mean() for _ in range(B)])
    return dict(cv2=cv2, n=n, mean_cv2=cv2.mean(), boot=boot, n_days=len(dv))


def rmse_true(tc, k, q=None):
    if q is None:
        return 100 * np.sqrt(tc['mean_cv2'] / k)
    return np.percentile(100 * np.sqrt(tc['boot'] / k), q)


def rmse_formula(tc, k, kind):
    ok = tc['n'] >= (2 * k if kind == 'disjoint' else k)
    c, n = tc['cv2'][ok], tc['n'][ok]
    if kind == 'nested':
        return 100 * np.sqrt(np.mean(c * (1 / k - 1 / n)))
    return 100 * np.sqrt(np.mean(c * (1 / k + 1 / (n - k))))


def spread_curve(P, idx, ks, R=50, seed=5):
    """Spatially spread subsets (farthest-point selection from a random first site) versus random
    subsets; both compared with the mean of the complementary sites."""
    rng = np.random.default_rng(seed); dv = day_values(P, idx); rows = []
    for k in ks:
        es, er = [], []
        for d, names, v in dv:
            n = len(v)
            if n < 2 * k:
                continue
            ii = np.array([list(P.columns).index(c) for c in names]); Dm = DIST[np.ix_(ii, ii)]
            e_s, e_r = [], []
            for _ in range(R):
                sel = [int(rng.integers(n))]
                while len(sel) < k:
                    dmin = Dm[:, sel].min(1); dmin[sel] = -1
                    sel.append(int(np.argmax(dmin + rng.random(n) * 1e-6)))
                comp = np.setdiff1d(np.arange(n), sel)
                e_s.append((v[sel].mean() - v[comp].mean()) / v.mean())
                pr = rng.permutation(n)
                e_r.append((v[pr[:k]].mean() - v[pr[k:]].mean()) / v.mean())
            es.append(np.mean(np.square(e_s))); er.append(np.mean(np.square(e_r)))
        rows.append((k, 100 * np.sqrt(np.mean(er)), 100 * np.sqrt(np.mean(es))))
    out = pd.DataFrame(rows, columns=['k', 'random_vs_complement', 'spread_vs_complement'])
    out['ratio_spread_to_random'] = out.spread_vs_complement / out.random_vs_complement
    return out


# =============================================================================
# 8. DESIGN CURVES (dense and routine sets)
# =============================================================================
KS_D = list(range(1, 11)); KS_R = [1, 2, 3]
DC_D = design_curves(P_all, Dd.index, KS_D, seed=11)
DC_R = design_curves(P_all, Rd.index, KS_R, seed=12)
TM_D = truth_curve(P_all, Dd.index, seed=13)
TM_R = truth_curve(P_all, Rd.index, seed=14)
n_med_D = int(Dd.n_sites.median()); n_med_R = int(Rd.n_sites.median())


def curve_table(DC, TM, ks, label):
    rows = []
    for k in ks:
        t = DC[k]
        if not len(t):
            continue
        pn, bn, _ = _boot_ci(t.mse_nest, seed=k)
        pdj, bdj, sk = _boot_ci(t.mse_disj, seed=100 + k)
        rows.append(dict(set=label, k=k, n_days=len(t),
                         nested_relRMSE=100 * np.sqrt(t.mse_nest.mean()), nested_lo=pn[0], nested_hi=pn[1],
                         nested_formula=rmse_formula(TM, k, 'nested'),
                         disjoint_relRMSE=100 * np.sqrt(t.mse_disj.mean()), disjoint_lo=pdj[0], disjoint_hi=pdj[1],
                         disjoint_bca_lo=bdj[0], disjoint_bca_hi=bdj[1], dayMSE_skew=sk,
                         disjoint_formula=rmse_formula(TM, k, 'disjoint'),
                         truth_relRMSE=rmse_true(TM, k), truth_lo=rmse_true(TM, k, 2.5),
                         truth_hi=rmse_true(TM, k, 97.5),
                         within20_pct=100 * t.w20.mean(), within10_pct=100 * t.w10.mean(),
                         wetdry_pct=100 * t.det.mean(), mae_mm=t.mae_mm.mean()))
    return pd.DataFrame(rows)


CT_D = curve_table(DC_D, TM_D, KS_D, 'dense')
CT_R = curve_table(DC_R, TM_R, KS_R, 'routine')
note('\n=== Design curves: relative RMSE (%) against three references ===')
note('dense set (%d days, median %d sites); routine set (%d days, median %d sites)'
     % (len(Dd), n_med_D, len(Rd), n_med_R))
note(pd.concat([CT_D, CT_R]).round(1).to_string(index=False))
for nm, TM, nmed in [('dense', TM_D, n_med_D), ('routine', TM_R, n_med_R)]:
    note('%s set: RMS spatial CV = %.3f (95%% CI %.3f-%.3f) -> single gauge vs true areal mean %.1f%%; '
         'error of a %d-site network mean %.1f%%'
         % (nm, np.sqrt(TM['mean_cv2']), *np.sqrt(np.percentile(TM['boot'], [2.5, 97.5])),
            rmse_true(TM, 1), nmed, rmse_true(TM, nmed)))
TRUE_K = pd.DataFrame({'k': range(1, 41)})
for nm, TM in [('dense', TM_D), ('routine', TM_R)]:
    TRUE_K[nm] = [rmse_true(TM, k) for k in TRUE_K.k]
    TRUE_K[nm + '_lo'] = [rmse_true(TM, k, 2.5) for k in TRUE_K.k]
    TRUE_K[nm + '_hi'] = [rmse_true(TM, k, 97.5) for k in TRUE_K.k]
SPREAD = spread_curve(P_all, Dd.index, KS_D)
note('spatially spread vs random subsets (dense set, both against the complementary sites):\n'
     + SPREAD.round(2).to_string(index=False))

# =============================================================================
# 9. DISTRIBUTION-FREE TOLERANCE (conformal) WITH EMPIRICAL COVERAGE
#    Score = |k-site mean - mean of the complementary sites| / areal mean.
#    The score involves no fitted model, so every calibration day is used;
#    for exchangeable days the radius covers a new day with probability >= 1-alpha.
# =============================================================================
def conf_q(s, alpha=ALPHA):
    s = np.sort(np.asarray(s)); n = len(s); r = int(np.ceil((n + 1) * (1 - alpha)))
    return s[r - 1] if r <= n else np.inf


def score_matrix(P, idx, k, m=None, R=200, seed=0):
    """R x n_days matrix of conformal scores, one random disjoint draw per day per repetition."""
    rng = np.random.default_rng(seed); dv = day_values(P, idx); cols = []
    for d, names, v in dv:
        mm = (len(v) - k) if m is None else m
        est, ref = draws_km(v, k, mm, R, rng)
        cols.append(np.abs(est - ref) / v.mean())
    return np.array(cols).T, [d for d, _, _ in dv]


def conformal_block(P, idx, ks, label, m=None, nmed=None, R=200, seed=200):
    """Marginal 90% radius plus checks that do not hold by construction:
    chronological split (calibrate on one half of the record, test on the other),
    coverage among light and heavy days (radius from the other days), and
    class-conditional (Mondrian) radii for light and heavy days."""
    rows = []
    for k in ks:
        Sm, dates = score_matrix(P, idx, k, m, R=R, seed=seed + k)
        dates = pd.DatetimeIndex(np.array(dates)); n = len(dates)
        mean_ = P.loc[dates].mean(1).values
        heavy = mean_ >= np.median(mean_)
        order = np.argsort(dates.values); first = np.zeros(n, bool); first[order[:n // 2]] = True
        rad, chrono, cl, ch, rl, rh = [], [], [], [], [], []
        for s in Sm:
            rad.append(conf_q(s))
            chrono.append(0.5 * (np.mean(s[~first] <= conf_q(s[first])) + np.mean(s[first] <= conf_q(s[~first]))))
            # s_i is covered by the radius computed from the other n-1 days
            # exactly when fewer than ceil(n(1-alpha)) of the others are smaller than s_i
            loo = stats.rankdata(s, method='min') <= int(np.ceil(n * (1 - ALPHA)))
            cl.append(loo[~heavy].mean()); ch.append(loo[heavy].mean())
            rl.append(conf_q(s[~heavy])); rh.append(conf_q(s[heavy]))
        rad = np.array(rad)
        ratio = np.sqrt((nmed - k) / nmed) if (m is None and nmed) else np.nan
        rows.append(dict(set=label, k=k, n_days=n, radius_pct=100 * rad.mean(),
                         radius_p05=100 * np.percentile(rad, 5), radius_p95=100 * np.percentile(rad, 95),
                         approx_truth_radius_pct=100 * rad.mean() * ratio,
                         chrono_coverage_pct=100 * np.mean(chrono),
                         coverage_light_pct=100 * np.mean(cl), coverage_heavy_pct=100 * np.mean(ch),
                         mondrian_radius_light_pct=100 * np.mean(rl), mondrian_radius_heavy_pct=100 * np.mean(rh),
                         median_areal_mean_mm=np.median(mean_)))
    return pd.DataFrame(rows)


CONF = conformal_block(P_all, Dd.index, KS_D, 'dense', nmed=n_med_D, seed=200)
CONF_R = conformal_block(P_all, Rd.index, KS_R, 'routine', nmed=n_med_R, seed=250)
note('\n=== 90% distribution-free tolerance radius (reference = complementary sites) ===')
note(pd.concat([CONF, CONF_R]).round(1).to_string(index=False))

TARGETS = [50, 30, 20, 10]
mins = []
for tgt in TARGETS:
    a = CONF.loc[CONF.radius_pct <= tgt, 'k']; b = CONF.loc[CONF.approx_truth_radius_pct <= tgt, 'k']
    c = [k for k in range(1, 41) if rmse_true(TM_D, k) <= tgt]
    mins.append(dict(target_pct=tgt, conformal_complement=int(a.iloc[0]) if len(a) else '>10',
                     approx_truth_radius=int(b.iloc[0]) if len(b) else '>10',
                     truth_rmse=c[0] if c else '>40'))
MINS = pd.DataFrame(mins)
note('minimum number of sites for a target accuracy (dense set):\n' + MINS.to_string(index=False))

# ---- exchangeability diagnostics ----
def runs_test(x):
    x = np.asarray(x); b = x > np.median(x); n1, n2 = b.sum(), (~b).sum()
    runs = 1 + np.sum(b[1:] != b[:-1])
    mu = 2 * n1 * n2 / (n1 + n2) + 1
    var = 2 * n1 * n2 * (2 * n1 * n2 - n1 - n2) / ((n1 + n2) ** 2 * (n1 + n2 - 1))
    z = (runs - mu) / np.sqrt(var) if var > 0 else 0
    return 2 * (1 - stats.norm.cdf(abs(z)))


exch = []
for nm, idx, k, m in [('dense', Dd.index, 1, None), ('dense', Dd.index, 3, None),
                      ('routine', Rd.index, 1, 4), ('routine', Rd.index, 3, 4)]:
    Sm, dates = score_matrix(P_all, idx, k, m, R=200, seed=300 + k)
    sbar = Sm.mean(0)
    order = np.argsort(np.array(dates))
    s_o = sbar[order]
    rho = stats.spearmanr(s_o[:-1], s_o[1:]).correlation
    yrs = pd.DatetimeIndex(np.array(dates)).year
    kw = stats.kruskal(*[sbar[yrs == y] for y in np.unique(yrs)]).pvalue if len(np.unique(yrs)) > 1 else np.nan
    warm = pd.DatetimeIndex(np.array(dates)).month.isin([4, 5, 6, 7, 8, 9])
    mw = stats.mannwhitneyu(sbar[warm], sbar[~warm]).pvalue if (warm.sum() > 5 and (~warm).sum() > 5) else np.nan
    exch.append(dict(set=nm, k=k, reference='complement' if m is None else 'm=%d' % m, n_days=len(sbar),
                     lag1_spearman=rho, runs_test_p=runs_test(s_o), kruskal_year_p=kw, warm_vs_cold_p=mw))
EXCH = pd.DataFrame(exch)

# cross-period / cross-network coverage with a matched design (k sites vs 4 other sites)
xc = []
for k in [1, 2, 3]:
    SD_, dD = score_matrix(P_all, Dd.index, k, 4, R=200, seed=400 + k)
    SR_, dR = score_matrix(P_all, Rd.index, k, 4, R=200, seed=500 + k)
    yrsR = pd.DatetimeIndex(np.array(dR)).year
    warmR = pd.DatetimeIndex(np.array(dR)).month.isin([4, 5, 6, 7, 8, 9])
    xc.append(dict(k=k, test='calibrate routine -> test dense',
                   coverage_pct=100 * np.mean([np.mean(sd <= conf_q(sr)) for sd, sr in zip(SD_, SR_)])))
    xc.append(dict(k=k, test='calibrate dense -> test routine',
                   coverage_pct=100 * np.mean([np.mean(sr <= conf_q(sd)) for sd, sr in zip(SD_, SR_)])))
    xc.append(dict(k=k, test='routine: calibrate warm -> test cold season',
                   coverage_pct=100 * np.mean([np.mean(sr[~warmR] <= conf_q(sr[warmR])) for sr in SR_])))
    xc.append(dict(k=k, test='routine: calibrate cold -> test warm season',
                   coverage_pct=100 * np.mean([np.mean(sr[warmR] <= conf_q(sr[~warmR])) for sr in SR_])))
    for y in np.unique(yrsR):
        if (yrsR == y).sum() >= 10:
            xc.append(dict(k=k, test='routine: leave-out year %d' % y,
                           coverage_pct=100 * np.mean([np.mean(sr[yrsR == y] <= conf_q(sr[yrsR != y])) for sr in SR_])))
XCOV = pd.DataFrame(xc)
note('\n=== Exchangeability diagnostics of the conformal scores ===')
note(EXCH.round(3).to_string(index=False))
note('coverage of the 90% radius when calibration and test days come from different periods/networks:')
note(XCOV.pivot_table(index='test', columns='k', values='coverage_pct').round(1).to_string())

# =============================================================================
# 10. THIESSEN (AREA-WEIGHTED) REFERENCE
# =============================================================================
DOMAIN = HULL.buffer(300)


def _voronoi_finite(vor):
    new_regions = []; new_vertices = vor.vertices.tolist()
    center = vor.points.mean(axis=0); radius = np.ptp(vor.points, axis=0).max() * 2
    ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        ridges.setdefault(p1, []).append((p2, v1, v2)); ridges.setdefault(p2, []).append((p1, v1, v2))
    for p1, reg in enumerate(vor.point_region):
        verts = vor.regions[reg]
        if all(v >= 0 for v in verts):
            new_regions.append(verts); continue
        nr = [v for v in verts if v >= 0]
        for p2, v1, v2 in ridges[p1]:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                continue
            t = vor.points[p2] - vor.points[p1]; t /= np.linalg.norm(t); n = np.array([-t[1], t[0]])
            mid = vor.points[[p1, p2]].mean(axis=0)
            far = vor.vertices[v2] + np.sign(np.dot(mid - center, n)) * n * radius
            nr.append(len(new_vertices)); new_vertices.append(far.tolist())
        vs = np.asarray([new_vertices[v] for v in nr]); c = vs.mean(axis=0)
        nr = np.array(nr)[np.argsort(np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0]))]
        new_regions.append(nr.tolist())
    return new_regions, np.asarray(new_vertices)


_wcache = {}


def thiessen_weights(names):
    key = tuple(sorted(names))
    if key in _wcache:
        w = _wcache[key]
    else:
        pts = np.array([coord[c] for c in key])
        if len(key) < 4:
            w = np.full(len(key), 1 / len(key))
        else:
            try:
                regs, verts = _voronoi_finite(Voronoi(pts))
                w = np.array([max(Polygon(verts[r]).intersection(DOMAIN).area, 0.0) for r in regs])
                w = w / w.sum() if w.sum() > 0 else np.full(len(key), 1 / len(key))
            except Exception:
                w = np.full(len(key), 1 / len(key))
        _wcache[key] = w
    pos = {c: i for i, c in enumerate(key)}
    return np.array([w[pos[c]] for c in names])


rob = []
for nm, idx in [('dense', Dd.index), ('routine', Rd.index)]:
    for d, names, v in day_values(P_all, idx):
        rob.append((nm, d, v.mean(), float((v * thiessen_weights(list(names))).sum())))
ROB = pd.DataFrame(rob, columns=['set', 'date', 'unweighted', 'thiessen'])
ROB['rel_diff'] = (ROB.thiessen - ROB.unweighted) / ROB.unweighted
note('\n=== Thiessen vs unweighted areal reference ===')
for nm, g in ROB.groupby('set'):
    note('  %s: n=%d | r=%.4f | median rel. diff %+.1f%% | median |diff| %.1f%% | p90 |diff| %.1f%% | max |diff| %.1f%%'
         % (nm, len(g), g.unweighted.corr(g.thiessen), 100 * g.rel_diff.median(), 100 * g.rel_diff.abs().median(),
            100 * g.rel_diff.abs().quantile(.9), 100 * g.rel_diff.abs().max()))

rng_t = np.random.default_rng(21); thi = []
dvD = day_values(P_all, Dd.index)
for k in KS_D:
    eu, et = [], []
    for d, names, v in dvD:
        n = len(v); e_u, e_t = [], []
        for _ in range(30):
            perm = rng_t.permutation(n); s, c = perm[:k], perm[k:]
            est = v[s].mean(); Ru = v[c].mean(); Rt = float((v[c] * thiessen_weights(list(names[c]))).sum())
            e_u.append((est - Ru) / v.mean()); e_t.append((est - Rt) / v.mean())
        eu.append(np.mean(np.square(e_u))); et.append(np.mean(np.square(e_t)))
    thi.append((k, 100 * np.sqrt(np.mean(eu)), 100 * np.sqrt(np.mean(et))))
THI = pd.DataFrame(thi, columns=['k', 'disjoint_unweighted', 'disjoint_thiessen'])
THI['abs_diff_pp'] = THI.disjoint_thiessen - THI.disjoint_unweighted
THI['rel_diff_pct'] = 100 * THI.abs_diff_pp / THI.disjoint_unweighted
note('  dense-set disjoint curve under both references:\n' + THI.round(1).to_string(index=False))

# =============================================================================
# 11. VARIANCE-MEAN SCALING, ITS PREDICTORS, AND MODEL VALIDATION
# =============================================================================
VAR_MIN = 8
feat = []
for d in daily.index[daily.wet & (daily.n_sites >= VAR_MIN)]:
    v = P_all.loc[d].dropna().values
    if v.mean() <= WET_MM or v.std(ddof=1) == 0:
        continue
    doy = d.dayofyear
    feat.append((d, v.mean(), v.std(ddof=1), v.std(ddof=1) / v.mean(), v.max() - v.min(),
                 (v > RAIN_MM).mean(), len(v), np.sin(2 * np.pi * doy / 365.25),
                 np.cos(2 * np.pi * doy / 365.25)))
F = pd.DataFrame(feat, columns=['date', 'mean', 'sd', 'cv', 'range', 'wetfrac', 'n', 'sin', 'cos'])
lx, ly = np.log(F['mean']), np.log(F.sd)
b, a_ = np.polyfit(lx, ly, 1); r2 = np.corrcoef(lx, ly)[0, 1] ** 2
rb = np.random.default_rng(31); bexp = []
for _ in range(2000):
    ii = rb.integers(0, len(F), len(F)); bexp.append(np.polyfit(lx.values[ii], ly.values[ii], 1)[0])
note('\n=== Variance-mean scaling (days with >=%d sites, n=%d) ===' % (VAR_MIN, len(F)))
note('SD = %.2f * mean^%.2f (exponent 95%% CI %.2f-%.2f); log-log R2 = %.2f; CV ~ mean^%.2f'
     % (np.exp(a_), b, *np.percentile(bexp, [2.5, 97.5]), r2, b - 1))
for nm, g in [('dense', F[F.date.isin(Dd.index)]), ('routine', F[F.date.isin(Rd.index)])]:
    if len(g) > 5:
        bb = np.polyfit(np.log(g['mean']), np.log(g.sd), 1)[0]
        note('  %s days only: exponent %.2f (n=%d)' % (nm, bb, len(g)))

LABELS = {'log10_mean': 'Areal mean (log$_{10}$ mm)', 'wetfrac': 'Fraction of gauges wet',
          'n': 'Number of reporting gauges', 'sin': 'Season, sine of day of year',
          'cos': 'Season, cosine of day of year'}
X = F[['mean', 'wetfrac', 'n', 'sin', 'cos']].copy(); X['mean'] = np.log10(X['mean'])
X = X.rename(columns={'mean': 'log10_mean'}); y = np.log10(F.sd).values
yrs_F = F.date.dt.year.values
PARAMS = dict(n_estimators=400, num_leaves=15, learning_rate=0.03, min_child_samples=12,
              subsample=0.8, subsample_freq=1, verbose=-1)


def _r2(yt, yp):
    return 1 - np.sum((yt - yp) ** 2) / np.sum((yt - yt.mean()) ** 2)


def cv_scores(splits):
    out = {'power law (log mean only)': [], 'linear (all predictors)': [], 'gradient boosting (all predictors)': []}
    for tr_i, te_i in splits:
        Xtr, Xte, ytr, yte = X.iloc[tr_i], X.iloc[te_i], y[tr_i], y[te_i]
        c1 = np.polyfit(Xtr.log10_mean, ytr, 1); out['power law (log mean only)'].append((yte, np.polyval(c1, Xte.log10_mean)))
        A1 = np.c_[np.ones(len(Xtr)), Xtr.values]; c2 = np.linalg.lstsq(A1, ytr, rcond=None)[0]
        out['linear (all predictors)'].append((yte, np.c_[np.ones(len(Xte)), Xte.values] @ c2))
        mdl = lgb.LGBMRegressor(**PARAMS).fit(Xtr, ytr)
        out['gradient boosting (all predictors)'].append((yte, mdl.predict(Xte)))
    return out


cvrows = []
rk = np.random.default_rng(41)
for rep in range(10):
    perm = rk.permutation(len(X)); folds = np.array_split(perm, 5)
    res = cv_scores([(np.setdiff1d(perm, f), f) for f in folds])
    for mname, pairs in res.items():
        yt = np.concatenate([p[0] for p in pairs]); yp = np.concatenate([p[1] for p in pairs])
        cvrows.append(('5-fold CV (10 repeats)', mname, _r2(yt, yp)))
uy = np.unique(yrs_F)
if len(uy) > 1:
    res = cv_scores([(np.where(yrs_F != yy_)[0], np.where(yrs_F == yy_)[0]) for yy_ in uy])
    for mname, pairs in res.items():
        yt = np.concatenate([p[0] for p in pairs]); yp = np.concatenate([p[1] for p in pairs])
        cvrows.append(('leave-one-year-out', mname, _r2(yt, yp)))
mdl_full = lgb.LGBMRegressor(**PARAMS).fit(X, y)
cvrows.append(('in-sample (no hold-out)', 'gradient boosting (all predictors)', mdl_full.score(X, y)))
CVT = pd.DataFrame(cvrows, columns=['scheme', 'model', 'R2']).groupby(['scheme', 'model']).R2.agg(['mean', 'std']).reset_index()
note('model skill for log10(spatial SD):\n' + CVT.round(3).to_string(index=False))

# out-of-fold SHAP values (each day explained by a model that did not see it)
SV = np.zeros(X.shape); cnt = np.zeros(len(X))
for rep in range(5):
    perm = rk.permutation(len(X))
    for f in np.array_split(perm, 5):
        trn = np.setdiff1d(perm, f)
        m_ = lgb.LGBMRegressor(**PARAMS).fit(X.iloc[trn], y[trn])
        SV[f] += shap.TreeExplainer(m_).shap_values(X.iloc[f]); cnt[f] += 1
SV = SV / cnt[:, None]
imp = pd.Series(np.abs(SV).mean(0), index=X.columns)
grp = {'storm magnitude': imp['log10_mean'], 'season (sine + cosine)': np.abs(SV[:, 3] + SV[:, 4]).mean(),
       'fraction of gauges wet': imp['wetfrac'], 'number of reporting gauges': imp['n']}
note('out-of-fold mean |SHAP| (log10 units):  ' + ' | '.join('%s %.3f' % kv for kv in grp.items()))
note('Spearman correlation among predictors:\n' + X.corr(method='spearman').round(2).to_string())

# =============================================================================
# 12. INTER-GAUGE CORRELATION AND PAIRWISE ERROR
#     MSD = (mu1-mu2)^2 + (s1-s2)^2 + 2 s1 s2 (1-r)   (exact, population moments)
# =============================================================================
pairs = []
for i, g1 in enumerate(P_all.columns):
    for j in range(i + 1, len(P_all.columns)):
        g2 = P_all.columns[j]
        s = P_all[[g1, g2]].dropna(); s = s[(s[g1] > RAIN_MM) | (s[g2] > RAIN_MM)]
        if len(s) < 25:
            continue
        a, c = s[g1].values, s[g2].values
        mu1, mu2, s1, s2_ = a.mean(), c.mean(), a.std(), c.std()
        r = np.corrcoef(a, c)[0, 1]; msd = np.mean((a - c) ** 2); mu = s.values.mean()
        pairs.append((g1, g2, net_of[g1][0] + '-' + net_of[g2][0], DIST[i, j], len(s), r, np.sqrt(msd) / mu,
                      ((a > RAIN_MM) == (c > RAIN_MM)).mean(),
                      (mu1 - mu2) ** 2 / msd, (s1 - s2_) ** 2 / msd, 2 * s1 * s2_ * (1 - r) / msd))
PP = pd.DataFrame(pairs, columns=['g1', 'g2', 'pair_type', 'dist_m', 'n_days', 'r', 'nrmse', 'wetdry_agree',
                                  'share_bias', 'share_sd', 'share_corr'])
PP['one_minus_r'] = 1 - PP.r
edges = [0, 500, 1000, 1500, 2000, 3000, 4000, 7000]
PP['bin'] = pd.cut(PP.dist_m, edges)
BIN = PP.groupby('bin', observed=True).agg(d=('dist_m', 'median'), n_pairs=('r', 'size'), r=('r', 'median'),
                                           one_minus_r=('one_minus_r', 'median'), nrmse=('nrmse', 'median'),
                                           wetdry=('wetdry_agree', 'median'), share_bias=('share_bias', 'median'),
                                           share_sd=('share_sd', 'median'), share_corr=('share_corr', 'median')).reset_index()
rho_r = stats.spearmanr(PP.dist_m, PP.r).correlation
rho_e = stats.spearmanr(PP.dist_m, PP.nrmse).correlation
rho_re = stats.spearmanr(PP.one_minus_r, PP.nrmse).correlation
note('\n=== Inter-gauge correlation and pairwise error (%d pairs with >=25 common rainy days) ===' % len(PP))
note(BIN.round(3).to_string(index=False))
note('Spearman: distance-r %.2f | distance-NRMSE %.2f | (1-r)-NRMSE %.2f; pair types: %s'
     % (rho_r, rho_e, rho_re, PP.pair_type.value_counts().to_dict()))

# =============================================================================
# 13. SENSITIVITY OF THE MAIN RESULT TO QUALITY-CONTROL CHOICES
# =============================================================================
sens = []
for nm, P_ in [('primary (low flags removed)', P_all), ('no spatial screening', P_noscreen),
               ('low and high flags removed', P_strict)]:
    n_ = P_.notna().sum(1); m_ = P_.mean(1)
    idx = P_.index[(m_ > WET_MM) & (n_ >= DENSE_MIN)]
    tm = truth_curve(P_, idx, B=200, seed=13)
    dc = design_curves(P_, idx, [1, 5], R=200, seed=11)
    sens.append((nm, len(idx), rmse_true(tm, 1), rmse_true(tm, 5),
                 100 * np.sqrt(dc[1].mse_disj.mean()), 100 * np.sqrt(dc[5].mse_disj.mean())))
SENS = pd.DataFrame(sens, columns=['variant', 'dense_days', 'truth_k1', 'truth_k5', 'disjoint_k1', 'disjoint_k5'])
note('\n=== Sensitivity to quality-control choices (dense set, relative RMSE %) ===')
note(SENS.round(2).to_string(index=False))

# =============================================================================
# 14. SUB-DAILY ACCUMULATIONS (30-minute Princeton records, optional)
# =============================================================================
SUB = None; SUBM = None; CHK = None
if HAVE_30:
    m30 = pd.DataFrame(min30_rows)
    m30['station'] = pd.to_numeric(m30['STATION_ID'], errors='coerce')
    m30['rain'] = pd.to_numeric(m30['RAINFALL'], errors='coerce')
    m30['ldate'] = _fife_date(m30['LOCAL_OBS_DATE'])
    tdig = m30['LOCAL_OBS_TIME'].astype(str).str.replace(r'\D', '', regex=True)
    tdig = tdig.map(lambda t: t[:4] if len(t) > 4 else t.zfill(4))
    m30['minute'] = pd.to_numeric(tdig.str[:2], errors='coerce') * 60 + pd.to_numeric(tdig.str[2:], errors='coerce')
    m30 = m30.dropna(subset=['station', 'rain', 'ldate', 'minute'])
    m30 = m30[(m30.rain >= 0)]
    m30['station'] = m30.station.astype(int)
    note('\n=== 30-minute records ===')
    note('records %d | stations %d | period %s to %s | max 30-min value %.2f mm'
         % (len(m30), m30.station.nunique(), m30.ldate.min().date(), m30.ldate.max().date(), m30.rain.max()))
    # consistency with the daily archive (daily totals were aggregated in local time)
    d30 = m30.groupby(['station', 'ldate']).rain.sum()
    cmp_ = []
    for (stn, d), v in d30.items():
        if stn in S.columns and d in S.index and not np.isnan(S.at[d, stn]):
            cmp_.append((stn, d, v, S.at[d, stn]))
    CHK = pd.DataFrame(cmp_, columns=['station', 'date', 'sum_30min', 'daily'])
    CHK['abs_diff'] = (CHK.sum_30min - CHK.daily).abs()
    note('daily totals rebuilt from 30-min records vs daily archive: n=%d station-days | within 0.1 mm %.1f%% | '
         'median |diff| %.3f mm | max |diff| %.2f mm'
         % (len(CHK), 100 * (CHK.abs_diff <= 0.1).mean(), CHK.abs_diff.median(), CHK.abs_diff.max()))
    big30 = m30.sort_values('rain', ascending=False).head(5)
    note('five largest 30-min values:\n' + big30[['SITEGRID_ID', 'station', 'LOCAL_OBS_DATE', 'LOCAL_OBS_TIME', 'rain']].to_string(index=False))

    # station-day validity from the screened daily matrix; absent 30-min records on valid days = 0
    valid_site_day = P_all[PR].notna()
    DUR = [1, 3, 6, 24]
    sub_rows, sub_models = [], []
    for h in DUR:
        m30['blk'] = (m30.minute // (60 * h)).astype(int)
        agg = m30.groupby(['station', 'ldate', 'blk']).rain.sum().reset_index()
        agg['site'] = agg.station.map(site_of)
        agg = agg.dropna(subset=['site'])
        agg = agg.groupby(['site', 'ldate', 'blk']).rain.mean().reset_index()   # collocated -> site mean
        nb = 24 // h
        # all (site, day, block) combinations on days with >=15 valid Princeton sites
        recs = []
        for d in valid_site_day.index[valid_site_day.sum(1) >= 15]:
            vs = [c for c in PR if valid_site_day.at[d, c]]
            if len(vs) < 15:
                continue
            sub_ = agg[agg.ldate == d].pivot_table(index='blk', columns='site', values='rain', aggfunc='sum')
            sub_ = sub_.reindex(index=range(nb), columns=vs).fillna(0.0)
            for bk in range(nb):
                recs.append((d, bk, sub_.loc[bk].values.astype(float), vs))
        dv = [(d, bk, v) for d, bk, v, vs in recs if v.mean() > WET_MM]
        if len(dv) < 10:
            continue
        cv2 = np.array([v.var(ddof=1) / v.mean() ** 2 for _, _, v in dv])
        day_id = np.array([str(d.date()) for d, _, _ in dv]); udays = np.unique(day_id)
        rng_s = np.random.default_rng(600 + h); boot = []
        for _ in range(2000):
            pick = rng_s.choice(udays, len(udays), replace=True)
            boot.append(np.concatenate([cv2[day_id == u] for u in pick]).mean())
        boot = np.array(boot)
        # empirical check with disjoint random subsets (k sites vs the other sites)
        chk = []
        for k in (1, 3, 5):
            e = []
            for _, _, v in dv:
                if len(v) >= 2 * k:
                    est, ref = draws_km(v, k, len(v) - k, 100, rng_s)
                    e.append(np.mean(((est - ref) / v.mean()) ** 2))
            chk.append(100 * np.sqrt(np.mean(e)))
        sub_models.append(dict(duration_h=h, n_wet_intervals=len(dv), n_days=len(udays),
                               median_sites=int(np.median([len(v) for _, _, v in dv])),
                               rms_cv=np.sqrt(cv2.mean()), rms_cv_lo=np.sqrt(np.percentile(boot, 2.5)),
                               rms_cv_hi=np.sqrt(np.percentile(boot, 97.5)),
                               disjoint_k1=chk[0], disjoint_k3=chk[1], disjoint_k5=chk[2]))
        for k in range(1, 41):
            sub_rows.append(dict(duration_h=h, k=k, truth_relRMSE=100 * np.sqrt(cv2.mean() / k),
                                 lo=np.percentile(100 * np.sqrt(boot / k), 2.5),
                                 hi=np.percentile(100 * np.sqrt(boot / k), 97.5)))
    SUB = pd.DataFrame(sub_rows); SUBM = pd.DataFrame(sub_models)
    if len(SUB):
        note('\n=== Sub-daily accumulations (Princeton sites, days with >=15 valid sites) ===')
        note(SUBM.round(3).to_string(index=False))
        tabk = SUB[SUB.k.isin([1, 2, 3, 5, 8, 10, 15, 20])].pivot_table(index='k', columns='duration_h',
                                                                        values='truth_relRMSE')
        note('relative RMSE against the true areal mean (%):\n' + tabk.round(1).to_string())
        for tgt in [30, 20, 10]:
            note('  minimum sites for <=%d%%: ' % tgt + ' | '.join(
                '%dh: %s' % (h, (str(int(g.loc[g.truth_relRMSE <= tgt, 'k'].min())) if (g.truth_relRMSE <= tgt).any() else '>40'))
                for h, g in SUB.groupby('duration_h')))

# =============================================================================
# 15. SECOND NETWORK: WALNUT GULCH EXPERIMENTAL WATERSHED (optional)
#     Daily totals from the DAP aggregate report; gauge coordinates and the
#     watershed boundary from the SWRC GIS layers (UTM zone 12N, NAD83).
#     The DAP documentation states that zeros are used as placeholders for
#     missing values in multi-gauge queries; gauges are therefore treated as
#     missing outside their period of record and during the listed outages,
#     and isolated zeros among wet neighbours are screened by the same
#     neighbour check as for FIFE.
# =============================================================================
WG_RADIUS = 2000.0      # neighbourhood radius (m) of the spatial check (gauge spacing ~1.1 km)
WG_MIN_RAIN_DAYS = 100  # gauges with fewer rain days in the record are not used
MONSOON = [7, 8, 9]     # July-September: North American Monsoon thunderstorms
WG = {}


def _find(pattern):
    hits = [f for f in glob.glob(os.path.join(WORKDIR, '*')) if re.search(pattern, os.path.basename(f), re.I)]
    return sorted(hits)


def _wg_report():
    for f in glob.glob(os.path.join(WORKDIR, '*')):
        if f.lower().endswith(ARCH_EXT) or os.path.getsize(f) < 1000:
            continue
        try:
            head = open(f, encoding='latin-1').read(4000)
        except Exception:
            continue
        if 'Aggregate (daily) Report' in head and 'Walnut Gulch' in head:
            return f
    return None


def items_from(P, idx, cols=None, min_valid=0, label=''):
    """List of (date, cluster label, values) for the given days and gauges."""
    out = []
    for d in idx:
        v = (P.loc[d, cols] if cols is not None else P.loc[d]).dropna().values.astype(float)
        if len(v) >= max(min_valid, 4) and v.mean() > WET_MM:
            out.append((d, label, v))
    return out


def tc_items(items, B=2000, seed=0):
    """True-areal-mean error for random placement, bootstrap over days (cluster = date)."""
    cv2 = np.array([v.var(ddof=1) / v.mean() ** 2 for _, _, v in items])
    days = np.array([str(d.date()) for d, _, _ in items]); ud, inv_ = np.unique(days, return_inverse=True)
    ssum = np.bincount(inv_, weights=cv2); scnt = np.bincount(inv_).astype(float)
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(ud), (B, len(ud)))
    boot = ssum[pick].sum(1) / scnt[pick].sum(1)
    return dict(cv2=cv2, n=np.array([len(v) for _, _, v in items]), mean_cv2=cv2.mean(),
                boot=boot, n_days=len(ud), n_items=len(items))


def disj_items(items, ks, R=100, seed=0):
    rng = np.random.default_rng(seed); rows = []
    for k in ks:
        en, ed = [], []
        for _, _, v in items:
            n = len(v)
            if n < 2 * k:
                continue
            est, ref = draws_km(v, k, n - k, R, rng)
            en.append(np.mean(((est - v.mean()) / v.mean()) ** 2)); ed.append(np.mean(((est - ref) / v.mean()) ** 2))
        if en:
            rows.append(dict(k=k, n_items=len(en), nested_relRMSE=100 * np.sqrt(np.mean(en)),
                             disjoint_relRMSE=100 * np.sqrt(np.mean(ed))))
    return pd.DataFrame(rows)


def conf_items(items, ks, R=100, seed=0):
    rng = np.random.default_rng(seed); rows = []
    dates = pd.DatetimeIndex([d for d, _, _ in items]); n = len(items)
    mean_ = np.array([v.mean() for _, _, v in items]); heavy = mean_ >= np.median(mean_)
    first = dates <= dates.sort_values()[n // 2 - 1]
    nmed = int(np.median([len(v) for _, _, v in items]))
    for k in ks:
        S = np.empty((R, n))
        for i, (_, _, v) in enumerate(items):
            est, ref = draws_km(v, k, len(v) - k, R, rng)
            S[:, i] = np.abs(est - ref) / v.mean()
        rad, chrono, cl, ch, rl, rh = [], [], [], [], [], []
        for s in S:
            rad.append(conf_q(s))
            chrono.append(0.5 * (np.mean(s[~first] <= conf_q(s[first])) + np.mean(s[first] <= conf_q(s[~first]))))
            loo = stats.rankdata(s, method='min') <= int(np.ceil(n * (1 - ALPHA)))
            cl.append(loo[~heavy].mean()); ch.append(loo[heavy].mean())
            rl.append(conf_q(s[~heavy])); rh.append(conf_q(s[heavy]))
        rad = np.array(rad)
        rows.append(dict(k=k, n_items=n, radius_pct=100 * rad.mean(),
                         approx_truth_radius_pct=100 * rad.mean() * np.sqrt((nmed - k) / nmed),
                         chrono_coverage_pct=100 * np.mean(chrono), coverage_light_pct=100 * np.mean(cl),
                         coverage_heavy_pct=100 * np.mean(ch), mondrian_radius_light_pct=100 * np.mean(rl),
                         mondrian_radius_heavy_pct=100 * np.mean(rh)))
    return pd.DataFrame(rows)


def scaling_items(items):
    m = np.array([v.mean() for _, _, v in items]); sd = np.array([v.std(ddof=1) for _, _, v in items])
    ok = sd > 0
    b, a = np.polyfit(np.log(m[ok]), np.log(sd[ok]), 1)
    return b, np.exp(a), np.corrcoef(np.log(m[ok]), np.log(sd[ok]))[0, 1] ** 2, m, sd


wg_file = _wg_report()
wg_shp = _find(r'raingage_digital\.shp$'); wg_bnd = _find(r'^\d+_boundary\.shp$') or _find(r'boundary\.shp$')
if wg_file and wg_shp:
    import shapefile
    from shapely.geometry import shape as _shape, Point
    tabs = pd.read_html(wg_file)
    main = [t for t in tabs if 'Year' in t.columns][0]
    offt = [t for t in tabs if any('Turned Off' in str(c) for c in t.columns)]
    main['date'] = pd.to_datetime(dict(year=main.Year, month=main.Month, day=main.Day))
    GW = main.set_index('date').drop(columns=['Year', 'Month', 'Day'])
    GW.columns = [int(str(c).split()[-1]) for c in GW.columns]
    GW = GW.apply(pd.to_numeric, errors='coerce').sort_index()

    rd = shapefile.Reader(wg_shp[0])
    fld = [f[0] for f in rd.fields[1:]]
    gdf = pd.DataFrame(rd.records(), columns=fld)
    gdf = gdf.rename(columns={'ID': 'gauge', 'I_East': 'east', 'I_North': 'north'})[['gauge', 'east', 'north']]
    BOUND = None
    if wg_bnd:
        BOUND = _shape(shapefile.Reader(wg_bnd[0]).shape(0).__geo_interface__)

    # period of record and outages -> missing
    n_raw = GW.shape[1]
    WGQ = []
    for g in GW.columns:
        pos = GW.index[GW[g] > 0]
        if len(pos) == 0:
            GW[g] = np.nan; continue
        GW.loc[(GW.index < pos.min()) | (GW.index > pos.max()), g] = np.nan
    n_out = 0
    if offt:
        o = offt[0]; o.columns = ['gauge', 'off', 'on']
        for r in o.itertuples():
            g = int(r.gauge)
            if g in GW.columns:
                msk = (GW.index >= pd.to_datetime(r.off)) & (GW.index <= pd.to_datetime(r.on))
                n_out += int(GW.loc[msk, g].notna().sum()); GW.loc[msk, g] = np.nan
    raindays = (GW > 0).sum()
    keep = [g for g in GW.columns if g in set(gdf.gauge) and raindays[g] >= WG_MIN_RAIN_DAYS]
    WGQ.append(('gauges in the daily report', n_raw))
    WGQ.append(('gauges with coordinates in the GIS layer and >= %d rain days' % WG_MIN_RAIN_DAYS, len(keep)))
    WGQ.append(('gauge-days set to missing during listed outages', n_out))
    GW = GW[keep]
    gx = gdf.set_index('gauge').loc[keep]
    XYw = gx[['east', 'north']].values
    DISTW = np.sqrt(((XYw[:, None, :] - XYw[None, :, :]) ** 2).sum(-1))
    fl_w, fh_w, _ = neighbour_flags(GW.values.copy(), DISTW, WG_RADIUS)
    WGQ.append(('gauge-days flagged low by the neighbour check (removed)', int(fl_w.sum())))
    WGQ.append(('gauge-days flagged high by the neighbour check (retained)', int(fh_w.sum())))
    GW = GW.mask(pd.DataFrame(fl_w, index=GW.index, columns=GW.columns))
    inside = np.array([BOUND.contains(Point(x, y)) for x, y in XYw]) if BOUND is not None else np.ones(len(keep), bool)
    Dw = DISTW.copy(); np.fill_diagonal(Dw, np.inf)
    note('\n=== Walnut Gulch network ===')
    for k_, v_ in WGQ:
        note('  %-80s %d' % (k_, v_))
    note('record %s to %s; %d days with rain at one or more gauges'
         % (GW.index.min().date(), GW.index.max().date(), len(GW)))
    note('gauges used: %d (%d inside the watershed boundary); boundary area %s km2; '
         'nearest-neighbour distance median %.0f m (range %.0f-%.0f)'
         % (len(keep), int(inside.sum()), '%.1f' % (BOUND.area / 1e6) if BOUND is not None else 'n/a',
            np.median(Dw.min(1)), Dw.min(1).min(), Dw.min(1).max()))

    # ---- analysis units: whole network and FIFE-sized windows ----
    A_FIFE = span_n * span_e * 1e6                     # FIFE bounding-box area (m2)
    R_WIN = np.sqrt(A_FIFE / np.pi)
    cnt_in = (DISTW <= R_WIN).sum(1)
    order = np.argsort(-cnt_in, kind='stable'); centres = []
    for i in order:
        if cnt_in[i] >= 10 and all(np.hypot(*(XYw[i] - XYw[c])) >= 2 * R_WIN for c in centres):
            centres.append(i)
    WINS = [(j + 1, XYw[c], [keep[i] for i in np.where(DISTW[c] <= R_WIN)[0]]) for j, c in enumerate(centres)]
    note('FIFE-sized windows: radius %.2f km (area %.1f km2); %d non-overlapping windows with %s gauges'
         % (R_WIN / 1000, A_FIFE / 1e6, len(WINS), ', '.join(str(len(w[2])) for w in WINS)))

    nval = GW.notna().sum(1)
    whole_days = GW.index[(nval >= 0.8 * len(keep)) & (GW.mean(1) > WET_MM)]
    seas = {'monsoon (Jul-Sep)': lambda d: d.month in MONSOON, 'non-monsoon': lambda d: d.month not in MONSOON}
    UNITS = {}
    for sname, sf in seas.items():
        UNITS[('watershed', sname)] = items_from(GW, [d for d in whole_days if sf(d)], label='watershed')
        it = []
        for wid, c, cols in WINS:
            ok = GW.index[GW[cols].notna().sum(1) >= 0.8 * len(cols)]
            it += items_from(GW, [d for d in ok if sf(d)], cols=cols, min_valid=int(0.8 * len(cols)), label='w%d' % wid)
        UNITS[('window', sname)] = it

    WG_TRUE, WG_DC, WG_CONF, WG_SCALE = [], [], [], []
    for (scale, sname), items in UNITS.items():
        if len(items) < 20:
            continue
        tc = tc_items(items, seed=700)
        nmed = int(np.median(tc['n']))
        for k in range(1, 41):
            WG_TRUE.append(dict(scale=scale, season=sname, k=k, truth_relRMSE=rmse_true(tc, k),
                                lo=rmse_true(tc, k, 2.5), hi=rmse_true(tc, k, 97.5)))
        ks = list(range(1, 11)) if scale == 'window' else [1, 2, 3, 5, 8, 10, 15, 20, 30]
        dc = disj_items(items, ks, seed=710).assign(scale=scale, season=sname)
        dc['truth_relRMSE'] = [rmse_true(tc, k) for k in dc.k]
        WG_DC.append(dc)
        WG_CONF.append(conf_items(items, [1, 2, 3, 5] if scale == 'window' else [1, 3, 5, 10], seed=720)
                       .assign(scale=scale, season=sname))
        b_, a_w, r2_, _, _ = scaling_items(items)
        WG_SCALE.append(dict(scale=scale, season=sname, n_items=tc['n_items'], n_days=tc['n_days'],
                             median_gauges=nmed, rms_cv=np.sqrt(tc['mean_cv2']),
                             single_gauge_truth=rmse_true(tc, 1), sd_exponent=b_, sd_coef=a_w, loglog_r2=r2_,
                             median_cv=np.median([v.std(ddof=1) / v.mean() for _, _, v in items])))
    WG_TRUE = pd.DataFrame(WG_TRUE); WG_DC = pd.concat(WG_DC); WG_CONF = pd.concat(WG_CONF)
    WG_SCALE = pd.DataFrame(WG_SCALE)
    note('summary by spatial unit and season:\n' + WG_SCALE.round(3).to_string(index=False))
    note('design curves (relative RMSE %):\n' + WG_DC.round(1).to_string(index=False))
    note('90% tolerance radius vs complementary gauges and coverage checks:\n' + WG_CONF.round(1).to_string(index=False))

    # ---- Thiessen vs unweighted (whole network) ----
    if BOUND is not None:
        _wc = {}

        def thiessen_w(cols):
            key = tuple(cols)
            if key not in _wc:
                pts = gx.loc[list(cols), ['east', 'north']].values
                regs, verts = _voronoi_finite(Voronoi(pts))
                w = np.array([max(Polygon(verts[r]).intersection(BOUND).area, 0.0) for r in regs])
                _wc[key] = w / w.sum()
            return _wc[key]
        rr = []
        for d in whole_days:
            v = GW.loc[d].dropna()
            rr.append((d, v.mean(), float((v.values * thiessen_w(list(v.index))).sum())))
        WG_ROB = pd.DataFrame(rr, columns=['date', 'unweighted', 'thiessen'])
        WG_ROB['rel_diff'] = (WG_ROB.thiessen - WG_ROB.unweighted) / WG_ROB.unweighted
        note('Thiessen vs unweighted watershed mean: n=%d | r=%.4f | median |diff| %.1f%% | p90 |diff| %.1f%%'
             % (len(WG_ROB), WG_ROB.unweighted.corr(WG_ROB.thiessen), 100 * WG_ROB.rel_diff.abs().median(),
                100 * WG_ROB.rel_diff.abs().quantile(.9)))
    else:
        WG_ROB = None

    # ---- inter-gauge correlation and error by season ----
    pw = []
    gcols = list(GW.columns)
    for sname, sf in seas.items():
        Gs = GW[[sf(d) for d in GW.index]]
        Av = Gs.values
        for i in range(len(gcols)):
            for j in range(i + 1, len(gcols)):
                a, c = Av[:, i], Av[:, j]
                ok = ~np.isnan(a) & ~np.isnan(c) & ((a > RAIN_MM) | (c > RAIN_MM))
                if ok.sum() < 25:
                    continue
                a, c = a[ok], c[ok]
                r = np.corrcoef(a, c)[0, 1]; msd = np.mean((a - c) ** 2)
                pw.append((sname, DISTW[i, j], ok.sum(), r, np.sqrt(msd) / np.concatenate([a, c]).mean(),
                           ((a > RAIN_MM) == (c > RAIN_MM)).mean(), 2 * a.std() * c.std() * (1 - r) / msd))
    WG_PP = pd.DataFrame(pw, columns=['season', 'dist_m', 'n_days', 'r', 'nrmse', 'wetdry_agree', 'share_corr'])
    wedges = [0, 1000, 2000, 3000, 4000, 6000, 8000, 12000, 16000, 26000]
    WG_PP['bin'] = pd.cut(WG_PP.dist_m, wedges)
    WG_BIN = (WG_PP.groupby(['season', 'bin'], observed=True)
              .agg(d=('dist_m', 'median'), n_pairs=('r', 'size'), r=('r', 'median'), nrmse=('nrmse', 'median'),
                   wetdry=('wetdry_agree', 'median'), share_corr=('share_corr', 'median')).reset_index())
    note('pairwise statistics by season and distance:\n' + WG_BIN.round(3).to_string(index=False))

    # ---- minimum gauges for target accuracy, both networks ----
    mrows = []
    HEAVY_MM = 5.0
    sets_ = [('FIFE dense set (1987)', 'all wet days', TM_D), ('FIFE routine set', 'all wet days', TM_R)]
    for nm_, idx_ in [('FIFE dense set (1987)', Dd.index), ('FIFE routine set', Rd.index)]:
        it_ = [x for x in items_from(P_all, idx_) if x[2].mean() >= HEAVY_MM]
        if len(it_) >= 10:
            sets_.append((nm_, 'areal mean >= 5 mm', tc_items(it_, B=500, seed=731)))
    for (scale, sname), items in UNITS.items():
        if len(items) >= 20:
            sets_.append(('Walnut Gulch %s, %s' % (scale, sname), 'all wet days', tc_items(items, B=500, seed=730)))
            it_ = [x for x in items if x[2].mean() >= HEAVY_MM]
            if len(it_) >= 10:
                sets_.append(('Walnut Gulch %s, %s' % (scale, sname), 'areal mean >= 5 mm',
                              tc_items(it_, B=500, seed=732)))
    for nm, cls, tc in sets_:
        row = dict(setting=nm, days=cls, n_days=tc['n_days'], single_gauge=rmse_true(tc, 1),
                   single_gauge_lo=rmse_true(tc, 1, 2.5), single_gauge_hi=rmse_true(tc, 1, 97.5))
        for tgt in [30, 20, 10]:
            kk = [k for k in range(1, 201) if rmse_true(tc, k) <= tgt]
            row['k_for_%d%%' % tgt] = kk[0] if kk else '>200'
        mrows.append(row)
    MIN_ALL = pd.DataFrame(mrows)
    note('minimum number of randomly placed gauges for a target error against the true areal mean:\n'
         + MIN_ALL.round(1).to_string(index=False))
    WG = dict(GW=GW, gx=gx, inside=inside, WINS=WINS, R_WIN=R_WIN, UNITS=UNITS)
else:
    note('\n(Walnut Gulch files not supplied: second-network analysis skipped)')


# =============================================================================
# 16. FIGURES
# =============================================================================
def savefig(fig, name):
    fig.savefig(os.path.join(OUTDIR, name), dpi=300); plt.close(fig)


# F1 network map
fig, ax = plt.subplots(figsize=(5.4, 5.2))
for net, mk, c in [('LTER', 's', BLUE), ('Princeton', 'o', RED)]:
    s = sites[sites.network == net]
    ax.scatter(s.utm_e / 1000, s.utm_n / 1000, marker=mk, s=44, c=c, edgecolor='k', linewidth=.4, zorder=3,
               label='%s (%d sites)' % (net, len(s)))
cs_ = sites[sites.n_instruments > 1]
ax.scatter(cs_.utm_e / 1000, cs_.utm_n / 1000, s=150, facecolor='none', edgecolor='k', linewidth=1, zorder=4,
           label='Two collocated gauges')
hx, hy = HULL.exterior.xy
ax.plot(np.array(hx) / 1000, np.array(hy) / 1000, color=GREY, lw=.8, ls='--', label='Convex hull of the network')
ax.set_xlabel('UTM easting, zone 14N (km)'); ax.set_ylabel('UTM northing (km)')
ax.set_aspect('equal'); ax.legend(fontsize=7.5, loc='lower left')
savefig(fig, 'F01_network_map.png')

# F2 availability
fig, ax = plt.subplots(figsize=(9, 4.6))
Mv = P_all.notna().astype(float).T
ax.imshow(Mv.values, aspect='auto', cmap='Greys', interpolation='nearest', extent=[0, Mv.shape[1], Mv.shape[0], 0])
ax.set_yticks(np.arange(Mv.shape[0]) + .5)
ax.set_yticklabels(['%s (%s)' % (c, net_of[c][0]) for c in Mv.index], fontsize=5)
xt = np.linspace(0, Mv.shape[1] - 1, 10).astype(int)
ax.set_xticks(xt + .5); ax.set_xticklabels([P_all.index[i].strftime('%Y-%m') for i in xt], rotation=45)
ax.set_xlabel('Days with at least one record (chronological order)'); ax.grid(False)
savefig(fig, 'F02_availability.png')

# F3 concurrent sites by network
fig, ax = plt.subplots(figsize=(8.5, 3.0))
ax.stackplot(daily.index, daily.n_lter, daily.n_princeton, colors=[BLUE, RED], alpha=.75,
             labels=['LTER sites', 'Princeton sites'], step='mid')
ax.axhline(DENSE_MIN, ls='--', c='k', lw=.8); ax.axhline(ROUTINE_MIN, ls=':', c='k', lw=.8)
ax.set_ylabel('Sites reporting'); ax.set_xlabel('Date'); ax.legend(fontsize=8, loc='upper left')
savefig(fig, 'F03_concurrent_sites.png')

# F4 instrument comparison
fig, axs = plt.subplots(1, 3, figsize=(13, 4.0), constrained_layout=True)
if len(NETCMP):
    for kind, col in [('Princeton-Princeton', GREY), ('LTER-Princeton', PURPLE)]:
        for _, r in NETCMP[NETCMP.kind == kind].iterrows():
            x = Aov[[r.site_a, r.site_b]].dropna(); x = x[(x.iloc[:, 0] > RAIN_MM) | (x.iloc[:, 1] > RAIN_MM)]
            axs[0].scatter(x.iloc[:, 0], x.iloc[:, 1], s=9, alpha=.5, color=col)
    mx_ = np.nanmax(Aov.values)
    axs[0].plot([0, mx_], [0, mx_], 'k--', lw=.7)
    axs[0].set_xscale('symlog', linthresh=1); axs[0].set_yscale('symlog', linthresh=1)
    axs[0].legend(handles=[Line2D([], [], marker='o', ls='', color=PURPLE, label='LTER vs Princeton'),
                           Line2D([], [], marker='o', ls='', color=GREY, label='Princeton vs Princeton')], fontsize=7.5)
    axs[0].set_xlabel('Daily rainfall, first gauge (mm)'); axs[0].set_ylabel('Daily rainfall, second gauge (mm)')
    axs[0].set_title('(a) Gauge pairs within 500 m', fontsize=9.5)
    xl = np.arange(3); wd = .38
    for o, (kind, col) in enumerate([('LTER-Princeton', PURPLE), ('Princeton-Princeton', GREY)]):
        g = NETCMP[NETCMP.kind == kind]
        axs[1].bar(xl + (o - .5) * wd, [g.r_lagm1.median(), g.r_lag0.median(), g.r_lagp1.median()], wd,
                   color=col, label=kind.replace('-', ' vs '))
    axs[1].set_xticks(xl); axs[1].set_xticklabels(['-1 day', '0', '+1 day'])
    axs[1].set_xlabel('Lag applied to the second gauge'); axs[1].set_ylabel('Median Pearson r')
    axs[1].set_ylim(0, 1); axs[1].legend(fontsize=7.5); axs[1].set_title('(b) Day-boundary check', fontsize=9.5)
for _, r in COLL.iterrows():
    x = S[[r.station_a, r.station_b]].dropna(); x = x[(x.iloc[:, 0] > RAIN_MM) | (x.iloc[:, 1] > RAIN_MM)]
    axs[2].scatter(x.iloc[:, 0], x.iloc[:, 1], s=12, alpha=.7, label='%s (%.0f m apart)' % (r.sitegrid, r.separation_m))
if len(COLL):
    mx_ = np.nanmax(S[list(COLL.station_a) + list(COLL.station_b)].values)
    axs[2].plot([0, mx_], [0, mx_], 'k--', lw=.7); axs[2].legend(fontsize=7.5)
axs[2].set_xlabel('Daily rainfall, gauge 1 (mm)'); axs[2].set_ylabel('Daily rainfall, gauge 2 (mm)')
axs[2].set_title('(c) Collocated tipping-bucket gauges', fontsize=9.5)
savefig(fig, 'F04_instrument_comparison.png')

# F5 design curve
fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.3), constrained_layout=True)
c = CT_D
ax[0].plot(c.k, c.nested_relRMSE, 'o:', color=GREY, label='Nested reference (all sites)')
ax[0].fill_between(c.k, c.disjoint_lo, c.disjoint_hi, color=BLUE, alpha=.15)
ax[0].plot(c.k, c.disjoint_relRMSE, 's-', color=BLUE, label='Disjoint reference (other sites)')
tk = TRUE_K[TRUE_K.k <= 12]
ax[0].fill_between(tk.k, tk.dense_lo, tk.dense_hi, color=RED, alpha=.15)
ax[0].plot(tk.k, tk.dense, '^-', color=RED, label='True areal mean, random placement')
ax[0].plot(SPREAD.k, SPREAD.spread_vs_complement, 'd--', color=GREEN, label='Spread placement, disjoint reference')
ax[0].plot(tk.k, tk.routine, 'v:', color=ORANGE, label='True areal mean, routine set')
ax[0].set_xlabel('Number of gauges, k'); ax[0].set_ylabel('Relative RMSE of the areal mean (%)')
ax[0].set_title('(a) Error against three references', fontsize=9.5); ax[0].legend(fontsize=7); ax[0].set_ylim(0, None)
ax[1].fill_between(CONF.k, CONF.radius_p05, CONF.radius_p95, color=BLUE, alpha=.15)
ax[1].plot(CONF.k, CONF.radius_pct, 's-', color=BLUE, label='Distribution-free radius (disjoint reference)')
ax[1].plot(CONF.k, CONF.approx_truth_radius_pct, '^--', color=RED, label='Approximate radius, true areal mean')
for tgt in [30, 20]:
    ax[1].axhline(tgt, color='k', lw=.5, ls=':')
ax[1].set_xlabel('Number of gauges, k'); ax[1].set_ylabel('90% tolerance radius (%)')
ax[1].set_title('(b) 90% tolerance radius', fontsize=9.5); ax[1].legend(fontsize=7); ax[1].set_ylim(0, None)
wb = np.arange(len(CONF)); w_ = .27
ax[2].bar(wb - w_, CONF.coverage_light_pct, w_, color=ORANGE, label='Light days (below median)')
ax[2].bar(wb, CONF.coverage_heavy_pct, w_, color=BLUE, label='Heavy days (above median)')
ax[2].bar(wb + w_, CONF.chrono_coverage_pct, w_, color=GREY, label='Chronological split')
ax[2].axhline(100 * (1 - ALPHA), color=RED, ls='--', lw=1, label='Nominal 90%')
ax[2].set_xticks(wb); ax[2].set_xticklabels(CONF.k.astype(str))
ax[2].set_ylim(50, 101); ax[2].set_xlabel('Number of gauges, k'); ax[2].set_ylabel('Empirical coverage (%)')
ax[2].set_title('(c) Coverage of the 90% radius', fontsize=9.5); ax[2].legend(fontsize=6.5, loc='lower left')
savefig(fig, 'F05_design_curve.png')

# F6 variability (scaling, CV, absolute spread)
fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.0), constrained_layout=True)
xx = np.logspace(np.log10(F['mean'].min()), np.log10(F['mean'].max()), 50)
isD = F.date.isin(Dd.index).values
for a_x, yv, lab in [(ax[0], F.sd, 'Spatial standard deviation (mm)'), (ax[1], F.cv, 'Spatial coefficient of variation (-)'),
                     (ax[2], F['range'], 'Maximum minus minimum across gauges (mm)')]:
    a_x.scatter(F['mean'][~isD], yv[~isD], s=20, color=BLUE, edgecolor='k', linewidth=.2, alpha=.8, label='Routine days')
    a_x.scatter(F['mean'][isD], yv[isD], s=26, marker='^', color=RED, edgecolor='k', linewidth=.2, label='Dense days')
    a_x.set_xscale('log'); a_x.set_xlabel('Areal-mean rainfall (mm)'); a_x.set_ylabel(lab)
ax[0].plot(xx, np.exp(a_) * xx ** b, color='k', lw=1.4, label='SD = %.2f mean$^{%.2f}$' % (np.exp(a_), b))
ax[0].set_yscale('log'); ax[0].legend(fontsize=7.5); ax[0].set_title('(a) Variance-mean scaling', fontsize=9.5)
ax[1].plot(xx, np.exp(a_) * xx ** (b - 1), color='k', lw=1.4, label='CV ~ mean$^{%.2f}$' % (b - 1))
ax[1].legend(fontsize=7.5); ax[1].set_title('(b) Relative variability', fontsize=9.5)
ax[2].plot([.5, F['mean'].max()], [.5, F['mean'].max()], 'k--', lw=.7, label='1:1')
ax[2].set_yscale('log'); ax[2].legend(fontsize=7.5); ax[2].set_title('(c) Absolute spread', fontsize=9.5)
savefig(fig, 'F06_variability_scaling.png')

# F7 SHAP (out-of-fold), readable labels
fig = plt.figure(figsize=(7.5, 4.0))
shap.summary_plot(SV, X.rename(columns=LABELS), show=False, plot_size=None)
plt.xlabel('SHAP value (effect on log$_{10}$ spatial SD)')
plt.tight_layout(); plt.savefig(os.path.join(OUTDIR, 'F07_shap_out_of_fold.png'), dpi=300); plt.close('all')

# F8 heaviest three dense days
top = Dd.sort_values('areal_mean', ascending=False).head(3).index
fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.0), constrained_layout=True)
for ax_, d in zip(axes, top):
    v = P_all.loc[d].dropna()
    x = np.array([coord[i][0] / 1000 for i in v.index]); yv = np.array([coord[i][1] / 1000 for i in v.index])
    mk = np.array([net_of[i] == 'LTER' for i in v.index])
    sc = ax_.scatter(x[~mk], yv[~mk], c=v.values[~mk], s=150, cmap='YlGnBu', edgecolor='k', linewidth=.6,
                     zorder=3, vmin=v.min(), vmax=v.max())
    ax_.scatter(x[mk], yv[mk], c=v.values[mk], s=150, marker='s', cmap='YlGnBu', edgecolor='k', linewidth=.6,
                zorder=3, vmin=v.min(), vmax=v.max())
    cb = fig.colorbar(sc, ax=ax_, fraction=0.047, pad=0.02); cb.set_label('Daily rainfall (mm)', fontsize=8.5)
    ax_.set_title('%s (%d sites)\nareal mean %.0f mm, range %.0f-%.0f mm, CV %.2f'
                  % (d.strftime('%d %b %Y'), len(v), v.mean(), v.min(), v.max(), v.std() / v.mean()), fontsize=9.5)
    ax_.set_xlabel('UTM easting (km)', fontsize=8.5); ax_.set_aspect('equal'); ax_.margins(0.18)
axes[0].set_ylabel('UTM northing (km)', fontsize=8.5)
savefig(fig, 'F08_largest_dense_days.png')

# F9 pairwise agreement
fig, axs = plt.subplots(2, 2, figsize=(10, 7.4), constrained_layout=True)
cols = {'L-L': BLUE, 'P-P': RED, 'L-P': PURPLE, 'P-L': PURPLE}
for a_x, col, lab in [(axs[0, 0], 'r', 'Pearson r on rainy days'), (axs[0, 1], 'nrmse', 'Pairwise normalised RMSD (-)'),
                      (axs[1, 0], 'wetdry_agree', 'Wet/dry agreement (-)')]:
    for t, g in PP.groupby('pair_type'):
        a_x.scatter(g.dist_m, g[col], s=10, alpha=.4, color=cols.get(t, GREY), edgecolor='none')
    a_x.plot(BIN.d, BIN[{'r': 'r', 'nrmse': 'nrmse', 'wetdry_agree': 'wetdry'}[col]], 'o-', color='k', ms=4)
    a_x.set_xlabel('Gauge separation (m)'); a_x.set_ylabel(lab)
axs[0, 0].legend(handles=[Line2D([], [], marker='o', ls='', color=BLUE, label='LTER pairs'),
                          Line2D([], [], marker='o', ls='', color=RED, label='Princeton pairs'),
                          Line2D([], [], marker='o', ls='', color=PURPLE, label='Mixed pairs'),
                          Line2D([], [], marker='o', color='k', label='Bin median')], fontsize=7)
ax2 = axs[0, 0].twinx(); ax2.plot(BIN.d, BIN.one_minus_r, 's--', color=ORANGE, ms=4)
ax2.set_yscale('log'); ax2.set_ylabel('1 - r (bin median, log scale)', color=ORANGE); ax2.grid(False)
axs[0, 0].set_title('(a) Correlation', fontsize=9.5); axs[0, 1].set_title('(b) Pairwise error', fontsize=9.5)
axs[1, 0].set_title('(c) Rain occurrence', fontsize=9.5)
wb = np.arange(len(BIN))
axs[1, 1].bar(wb, BIN.share_corr, color=BLUE, label='2 s$_1$ s$_2$ (1 - r)')
axs[1, 1].bar(wb, BIN.share_sd, bottom=BIN.share_corr, color=GREEN, label='(s$_1$ - s$_2$)$^2$')
axs[1, 1].bar(wb, BIN.share_bias, bottom=BIN.share_corr + BIN.share_sd, color=ORANGE, label='(mean$_1$ - mean$_2$)$^2$')
axs[1, 1].set_xticks(wb); axs[1, 1].set_xticklabels(['%.1f' % (x / 1000) for x in BIN.d])
axs[1, 1].set_xlabel('Median separation of distance bin (km)'); axs[1, 1].set_ylabel('Median share of mean squared difference')
axs[1, 1].legend(fontsize=7); axs[1, 1].set_title('(d) Decomposition of pairwise error', fontsize=9.5)
savefig(fig, 'F09_pairwise_agreement.png')

# F10 Thiessen reference
fig, ax = plt.subplots(1, 2, figsize=(9.8, 4.1), constrained_layout=True)
for nm, col, mk in [('routine', BLUE, 'o'), ('dense', RED, '^')]:
    g = ROB[ROB.set == nm]
    ax[0].scatter(g.unweighted, g.thiessen, s=20, color=col, marker=mk, edgecolor='k', linewidth=.2, alpha=.8,
                  label='%s days (r = %.3f)' % (nm.capitalize(), g.unweighted.corr(g.thiessen)))
mx = ROB[['unweighted', 'thiessen']].values.max()
ax[0].plot([0, mx], [0, mx], 'k--', lw=.7); ax[0].legend(fontsize=7.5)
ax[0].set_xlabel('Unweighted areal mean (mm)'); ax[0].set_ylabel('Thiessen areal mean (mm)')
ax[0].set_title('(a) Areal references', fontsize=9.5)
ax[1].plot(THI.k, THI.disjoint_unweighted, 'o-', color=BLUE, label='Unweighted reference')
ax[1].plot(THI.k, THI.disjoint_thiessen, 's--', color=RED, label='Thiessen reference')
ax[1].set_xlabel('Number of gauges, k'); ax[1].set_ylabel('Relative RMSE, disjoint reference (%)')
ax[1].set_title('(b) Dense-set design curve', fontsize=9.5); ax[1].legend(fontsize=7.5); ax[1].set_ylim(0, None)
savefig(fig, 'F10_reference_robustness.png')

# F11 sub-daily
if SUB is not None and len(SUB):
    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    pal = {1: RED, 3: ORANGE, 6: GREEN, 24: BLUE}
    for h, g in SUB.groupby('duration_h'):
        ax.fill_between(g.k, g.lo, g.hi, color=pal.get(h, GREY), alpha=.12)
        ax.plot(g.k, g.truth_relRMSE, '-', color=pal.get(h, GREY), label='%d h' % h)
    for tgt in [20, 10]:
        ax.axhline(tgt, color='k', lw=.5, ls=':')
    ax.set_yscale('log'); ax.set_xlabel('Number of gauges, k')
    ax.set_ylabel('Relative RMSE against the true areal mean (%)'); ax.legend(title='Accumulation', fontsize=8)
    savefig(fig, 'F11_subdaily_design_curves.png')

# F12-F13 Walnut Gulch
if WG:
    gx, inside = WG['gx'], WG['inside']
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    if BOUND is not None:
        bx, by = BOUND.exterior.xy
        ax.plot(np.array(bx) / 1000, np.array(by) / 1000, color='k', lw=.9, label='Watershed boundary')
    ax.scatter(gx.east[inside] / 1000, gx.north[inside] / 1000, s=22, c=BLUE, edgecolor='k', lw=.3, zorder=3,
               label='Gauges inside the watershed')
    if (~inside).any():
        ax.scatter(gx.east[~inside] / 1000, gx.north[~inside] / 1000, s=22, c='white', edgecolor=BLUE, lw=.8,
                   zorder=3, label='Gauges just outside the boundary')
    th = np.linspace(0, 2 * np.pi, 200)
    for wid, c, cols in WG['WINS']:
        ax.plot((c[0] + WG['R_WIN'] * np.cos(th)) / 1000, (c[1] + WG['R_WIN'] * np.sin(th)) / 1000,
                color=RED, lw=1, ls='--')
        ax.text(c[0] / 1000, c[1] / 1000 + WG['R_WIN'] / 1000 + .25, 'W%d (%d)' % (wid, len(cols)),
                ha='center', fontsize=7.5, color=RED)
    ax.plot([], [], color=RED, ls='--', label='Windows of the FIFE network size')
    ax.set_aspect('equal'); ax.set_xlabel('UTM easting, zone 12N (km)'); ax.set_ylabel('UTM northing (km)')
    ax.legend(fontsize=7.5, loc='upper center', bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False)
    savefig(fig, 'F12_walnut_gulch_network.png')

    fig, axs = plt.subplots(1, 3, figsize=(14, 4.3), constrained_layout=True)
    tk = TRUE_K[TRUE_K.k <= 20]
    axs[0].plot(tk.k, tk.dense, 'k-', lw=2, label='FIFE, dense set (summer 1987)')
    axs[0].plot(tk.k, tk.routine, 'k:', lw=1.6, label='FIFE, routine set')
    sty = {('window', 'monsoon (Jul-Sep)'): (RED, '-'), ('window', 'non-monsoon'): (BLUE, '-'),
           ('watershed', 'monsoon (Jul-Sep)'): (RED, '--'), ('watershed', 'non-monsoon'): (BLUE, '--')}
    for (scale, sname), g in WG_TRUE.groupby(['scale', 'season']):
        g = g[g.k <= 20]; col, ls = sty[(scale, sname)]
        axs[0].fill_between(g.k, g.lo, g.hi, color=col, alpha=.10)
        axs[0].plot(g.k, g.truth_relRMSE, color=col, ls=ls,
                    label='Walnut Gulch %s, %s' % ('22 km$^2$ windows' if scale == 'window' else 'watershed', sname))
    for tgt in [20, 10]:
        axs[0].axhline(tgt, color='k', lw=.5, ls=':')
    axs[0].set_yscale('log'); axs[0].set_xlabel('Number of randomly placed gauges, k')
    axs[0].set_ylabel('Relative RMSE against the true areal mean (%)'); axs[0].legend(fontsize=6.5)
    axs[0].set_title('(a) Design curves', fontsize=9.5)
    for (scale, sname), items in WG['UNITS'].items():
        if scale != 'window' or not items:
            continue
        m_ = np.array([v.mean() for _, _, v in items]); cv_ = np.array([v.std(ddof=1) / v.mean() for _, _, v in items])
        col = RED if sname.startswith('monsoon') else BLUE
        axs[1].scatter(m_, cv_, s=5, alpha=.25, color=col, edgecolor='none', label='Walnut Gulch windows, ' + sname)
    axs[1].scatter(F['mean'], F.cv, s=14, color='k', marker='^', label='FIFE')
    axs[1].set_xscale('log'); axs[1].set_yscale('log'); axs[1].set_xlabel('Areal-mean rainfall (mm)')
    axs[1].set_ylabel('Spatial coefficient of variation (-)'); axs[1].legend(fontsize=6.5, markerscale=2)
    axs[1].set_title('(b) Relative spatial variability', fontsize=9.5)
    for sname, g in WG_BIN.groupby('season'):
        col = RED if sname.startswith('monsoon') else BLUE
        axs[2].plot(g.d / 1000, g.r, 'o-', color=col, label='Walnut Gulch, ' + sname)
    axs[2].plot(BIN.d / 1000, BIN.r, 's-', color='k', label='FIFE')
    axs[2].set_xlabel('Gauge separation (km)'); axs[2].set_ylabel('Median Pearson r on rainy days')
    axs[2].legend(fontsize=7); axs[2].set_title('(c) Inter-gauge correlation', fontsize=9.5)
    savefig(fig, 'F13_network_comparison.png')

# =============================================================================
# 17. EXPORT
# =============================================================================
def _csv(obj, name, **kw):
    if obj is not None and len(obj):
        obj.to_csv(os.path.join(OUTDIR, name), **kw)


_csv(inv, 'T01_gauge_inventory.csv', index=False)
_csv(sites, 'T02_sites.csv', index=False)
_csv(COMP, 'T03_daily_composition_summary.csv', index=False)
_csv(daily[daily['set'] != ''], 'T04_daily_composition_by_day.csv')
_csv(pd.DataFrame(qc_log, columns=['check', 'count']), 'T05_qc_summary.csv', index=False)
_csv(FLAGS, 'T06_qc_flagged_site_days.csv', index=False)
_csv(RATIO, 'T07_site_to_neighbour_ratio.csv', index=False)
_csv(COLL, 'T08_collocated_gauges.csv', index=False)
_csv(NETCMP, 'T09_lter_vs_princeton_pairs.csv', index=False)
_csv(pd.concat([CT_D, CT_R]), 'T10_design_curves.csv', index=False)
_csv(TRUE_K, 'T11_true_areal_error_by_k.csv', index=False)
_csv(SPREAD, 'T11b_spread_vs_random_subsets.csv', index=False)
_csv(pd.concat([CONF, CONF_R]), 'T12_conformal_radius_coverage.csv', index=False)
_csv(MINS, 'T13_minimum_gauges.csv', index=False)
_csv(EXCH, 'T14_exchangeability.csv', index=False)
_csv(XCOV, 'T15_cross_period_coverage.csv', index=False)
_csv(ROB, 'T16_thiessen_vs_unweighted_daily.csv', index=False)
_csv(THI, 'T17_thiessen_design_curve.csv', index=False)
_csv(F, 'T18_daily_variability.csv', index=False)
_csv(CVT, 'T19_model_validation.csv', index=False)
_csv(pd.Series(grp, name='mean_abs_shap').to_frame(), 'T20_shap_grouped.csv')
_csv(PP.drop(columns=['bin']), 'T21_pairwise.csv', index=False)
_csv(BIN.astype({'bin': str}), 'T22_pairwise_binned.csv', index=False)
_csv(SENS, 'T23_qc_sensitivity.csv', index=False)
_csv(SUBM, 'T24_subdaily_models.csv', index=False)
_csv(SUB, 'T25_subdaily_curves.csv', index=False)
_csv(CHK, 'T26_daily_vs_30min_check.csv', index=False)
if WG:
    _csv(WG_TRUE, 'T27_wg_true_areal_error_by_k.csv', index=False)
    _csv(WG_DC, 'T28_wg_design_curves.csv', index=False)
    _csv(WG_CONF, 'T29_wg_conformal.csv', index=False)
    _csv(WG_SCALE, 'T30_wg_summary_by_unit.csv', index=False)
    _csv(WG_BIN.astype({'bin': str}), 'T31_wg_pairwise_binned.csv', index=False)
    _csv(WG_ROB, 'T32_wg_thiessen.csv', index=False)
    _csv(MIN_ALL, 'T33_minimum_gauges_both_networks.csv', index=False)
    _csv(pd.DataFrame([(w, c[0], c[1], len(cols), ' '.join(map(str, cols))) for w, c, cols in WG['WINS']],
                      columns=['window', 'centre_east', 'centre_north', 'n_gauges', 'gauges']),
         'T34_wg_windows.csv', index=False)
P_all.to_csv(os.path.join(OUTDIR, 'site_by_day_matrix.csv'))
with open(os.path.join(OUTDIR, 'results_summary.txt'), 'w') as fh:
    fh.write('Point-to-areal rainfall representativeness (FIFE/Konza and Walnut Gulch) - results\n\n')
    fh.write('\n'.join(SUMMARY))

_zip = shutil.make_archive('fife_full_outputs', 'zip', OUTDIR)
print('\nBundled -> %s (%.0f KB) | %d files' % (os.path.basename(_zip), os.path.getsize(_zip) / 1024,
                                                 len(os.listdir(OUTDIR))))
if IN_COLAB:
    from google.colab import files as _f
    _f.download('fife_full_outputs.zip')
