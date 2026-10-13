"""
=====================================================================
  SOMAIYASAT AI LINK MANAGER            (AI use case KJS-SRS-01)
  AI-based adaptive modulation for the satellite downlink
  Project : AI Integration in Digital Communication
=====================================================================
  Satellite : data queue (TT&C -> image -> voice). Every second the
              AI picks a mode: WAIT / BPSK / QPSK / 8-PSK, using the
              link estimate, battery level and what it is sending
  Channel   : distance (inverse-square law) + tumbling antenna + AWGN
  Ground    : demodulator -> packet check -> TT&C / image / voice
  The AI is a small decision tree trained on simulated passes.

  Run  : python somaiyasat_ai_link.py   (or paste into one Colab cell)
  Needs: numpy, scipy, matplotlib, scikit-learn  (pre-installed on Colab)
=====================================================================
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc
from sklearn.tree import DecisionTreeClassifier

rng = np.random.default_rng(7)
plt.rcParams['font.size'] = 13                       # readable when printed in the report

# ---------------- mission parameters (our assumptions) ----------------
MODES      = ['WAIT', 'BPSK', 'QPSK', '8PSK']
K          = [0, 1, 2, 3]            # bits per symbol (WAIT = transmitter off)
SYM_RATE   = 1024                    # symbols per second
PKT_BITS   = 1024                    # 128-byte packets, each checked on arrival
SNR_ZENITH = 22.0                    # downlink Es/N0 in dB, satellite overhead, full power
TARGET_PER = 0.01                    # AI goal: at least 99% of packets must arrive
R_EARTH    = 6371.0                  # km
ALT        = 500.0                   # orbit altitude, km
OMEGA      = 2*np.pi / 5668          # orbital angular speed, rad/s (94.5-minute orbit)
MIN_ELEV   = 5                       # ground station tracks above 5 degrees elevation
TUMBLE_DB  = 2.0                     # tumbling antenna makes the signal wobble +/-2 dB
EST_ERR_DB = 1.0                     # error of the satellite's link measurement (1 dB)
LOW_POWER  = 3.0                     # below 50% battery: half TX power = 3 dB less SNR
N_TTC, N_IMG = 16, 96                # packets queued: TT&C first, then 96 image rows,
                                     # then voice (M17/Codec2) fills the rest of the pass

# =================================================================
# 1. MODEM : Gray-coded BPSK / QPSK / 8-PSK over AWGN
# =================================================================
def constellation(mode):
    if mode == 'BPSK':
        return np.array([-1, 1], dtype=complex)
    if mode == 'QPSK':
        return np.array([-1-1j, -1+1j, 1-1j, 1+1j]) / np.sqrt(2)
    pts = np.zeros(8, dtype=complex)                      # 8-PSK, Gray code round the circle
    for k in range(8):
        pts[k ^ (k >> 1)] = np.exp(2j*np.pi*k/8)
    return pts

CONST = {m: constellation(m) for m in MODES[1:]}
NAME  = {'WAIT': 'WAIT', 'BPSK': 'BPSK', 'QPSK': 'QPSK', '8PSK': '8-PSK'}   # labels for plots

def modulate(bits, mode):
    k   = K[MODES.index(mode)]
    idx = bits.reshape(-1, k) @ (1 << np.arange(k)[::-1])        # k bits -> symbol index
    return CONST[mode][idx]

def demodulate(r, mode):                                       # minimum-distance detector
    k   = K[MODES.index(mode)]
    idx = np.argmin(np.abs(r[:, None] - CONST[mode][None, :]), axis=1)
    return ((idx[:, None] >> np.arange(k)[::-1]) & 1).astype(np.uint8).ravel()

def awgn(s, snr_db):                                           # Es = 1, SNR = Es/N0
    n0 = 10 ** (-snr_db / 10)
    return s + np.sqrt(n0/2) * (rng.standard_normal(s.size) + 1j*rng.standard_normal(s.size))

# =================================================================
# 2. TEXTBOOK THEORY : BER -> packet error rate -> best mode
# =================================================================
def Q(x):
    return 0.5 * erfc(x / np.sqrt(2))

def ber_theory(mode, snr_db):
    g = 10 ** (np.asarray(snr_db) / 10)
    if mode == 'BPSK':
        return Q(np.sqrt(2*g))
    if mode == 'QPSK':
        return Q(np.sqrt(g))
    return 2/3 * Q(np.sqrt(2*g) * np.sin(np.pi/8))              # 8-PSK (Gray, approx.)

def per(mode, snr_db):                                         # packet lost if any bit wrong
    return 1 - (1 - ber_theory(mode, snr_db)) ** PKT_BITS

def best_mode(snr_db, priority):
    # priority data (TT&C, image): fastest mode that still delivers >= 99% of packets
    # voice: mode that delivers the most correct bits (a lost packet = short glitch)
    snr_db = np.asarray(snr_db, dtype=float)
    safe = np.zeros(snr_db.shape, dtype=int)                   # 0 = WAIT
    for m in (1, 2, 3):
        safe[per(MODES[m], snr_db) <= TARGET_PER] = m
    goodput = np.stack([K[m] * (1 - per(MODES[m], snr_db)) for m in (1, 2, 3)])
    fast = 1 + np.argmax(goodput, axis=0)
    return np.where(priority, safe, fast)

# =================================================================
# 3. ONE SATELLITE PASS : geometry -> signal strength every second
# =================================================================
def make_pass(battery):
    t    = np.arange(-400, 401)                                # s from closest approach
    lam  = OMEGA * t                                           # Earth-centre angle
    d    = np.sqrt(R_EARTH**2 + (R_EARTH+ALT)**2
                   - 2*R_EARTH*(R_EARTH+ALT)*np.cos(lam))      # slant range (cosine rule)
    elev = np.degrees(np.arcsin(((R_EARTH+ALT)*np.cos(lam) - R_EARTH) / d))
    up   = elev >= MIN_ELEV
    d, elev = d[up], elev[up]
    t    = np.arange(elev.size)                                # s since pass start
    tumble = TUMBLE_DB*np.sin(2*np.pi*t/rng.uniform(20, 60) + rng.uniform(0, 2*np.pi))
    link = SNR_ZENITH - 20*np.log10(d/ALT) + tumble            # inverse-square law + tumbling
    true_snr = link - (LOW_POWER if battery < 50 else 0.0)     # low battery -> half TX power
    estimate = link + rng.normal(0, EST_ERR_DB, link.size)     # measured on ground beacon
    return dict(t=t, elev=elev, battery=battery, true_snr=true_snr, estimate=estimate)

def features(p, priority):   # what the satellite knows at each second
    n = p['t'].size
    return np.column_stack([p['estimate'], np.full(n, p['battery']), np.full(n, priority)])

FEATURE_NAMES = ['link_estimate_dB', 'battery_pct', 'sending_priority_data']

# =================================================================
# 4. AI LINK MANAGER : decision tree trained on simulated passes
# =================================================================
print("Simulating 300 training passes ...")
train = [make_pass(rng.uniform(20, 100)) for _ in range(300)]
X, y = [], []
for p in train:                                                # both kinds of data, every second
    for prio in (1, 0):
        X.append(features(p, prio))
        y.append(best_mode(p['true_snr'], prio))               # correct answer (known in simulation)
X, y = np.vstack(X), np.concatenate(y)
# depth 6: depth 3 was too simple (lost TT&C), depth 7 invented battery rules (overfitting)
ai = DecisionTreeClassifier(max_depth=6, min_samples_leaf=200, random_state=0).fit(X, y)

def switches(decide):                 # probe a policy at every link value from 0 to 30 dB
    grid = np.arange(0, 30, 0.01)
    pol = decide(grid)
    return f"{MODES[pol[0]]}" + ''.join(f", {MODES[pol[i]]} from {grid[i]:.1f}"
                                        for i in range(1, grid.size) if pol[i] != pol[i-1])

print(f"  {len(y)} training examples, tree with {ai.get_n_leaves()} end points.")
print("  Where the AI switches mode (link estimate, dB) vs textbook theory (+3 dB at low battery):")
for batt, blabel, shift in [(80, 'full power ', 0.0), (30, 'low battery', LOW_POWER)]:
    for prio, plabel in [(1, 'priority'), (0, 'voice   ')]:
        ai_rule = switches(lambda g: ai.predict(np.column_stack([g, np.full(g.size, batt), np.full(g.size, prio)])))
        theory  = switches(lambda g: best_mode(g - shift, prio))
        print(f"   {blabel} {plabel} | AI: {ai_rule}\n   {'':21s}| theory: {theory}")

# each strategy gives two plans: (mode when sending priority data, mode when sending voice)
STRATEGIES = {
    'Fixed BPSK'   : lambda p: (np.full(p['t'].size, 1),) * 2,
    'Fixed QPSK'   : lambda p: (np.full(p['t'].size, 2),) * 2,
    'Fixed 8-PSK'  : lambda p: (np.full(p['t'].size, 3),) * 2,
    'Textbook rule': lambda p: (best_mode(p['estimate'], 1),) * 2,   # BER thresholds only
    'AI'           : lambda p: (ai.predict(features(p, 1)), ai.predict(features(p, 0))),
    'Ideal'        : lambda p: (best_mode(p['true_snr'], 1), best_mode(p['true_snr'], 0)),
}

# =================================================================
# 5. DOWNLINK : priority queue, real bits through the modem, packet check
# =================================================================
def make_test_image():                                         # 96 x 128 grey image
    fig = plt.figure(figsize=(1.28, 0.96), dpi=100)
    ax  = fig.add_axes([0, 0, 1, 1])
    ax.imshow(np.tile(np.linspace(0.25, 0.75, 128), (96, 1)), cmap='gray',
              vmin=0, vmax=1, extent=[0, 128, 0, 96])
    ax.add_patch(plt.Rectangle((54, 44), 20, 20, color='white'))        # body
    ax.add_patch(plt.Rectangle((22, 49), 30, 10, color='black'))        # solar panels
    ax.add_patch(plt.Rectangle((76, 49), 30, 10, color='black'))
    ax.text(64, 20, 'SomaiyaSat', ha='center', va='center', fontsize=10,
            color='white', weight='bold')
    ax.set(xlim=(0, 128), ylim=(0, 96))
    ax.axis('off')
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[:, :, 0].copy()
    plt.close(fig)
    return img

IMAGE = make_test_image()

def packet_bits(i, battery):                                   # packet number -> 1024 bits
    if i < N_TTC:
        data = f'TT&C {i:02d} BATT={battery:.0f}% TEMP=21C MODE=NOMINAL'.ljust(128).encode()
    elif i < N_TTC + N_IMG:
        data = IMAGE[i - N_TTC].tobytes()                      # one image row per packet
    else:
        data = rng.integers(0, 256, 128, dtype=np.uint8).tobytes()   # voice (Codec2 frames)
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8))

def run_downlink(p, plans):
    prio_modes, voice_modes = plans
    nxt, received = 0, []
    per_slot, used = np.zeros(prio_modes.size), np.zeros(prio_modes.size, dtype=int)
    for slot in range(prio_modes.size):
        sending_priority = nxt < N_TTC + N_IMG                 # head of the priority queue
        m = prio_modes[slot] if sending_priority else voice_modes[slot]
        used[slot] = m
        if m == 0:
            continue                                           # WAIT: transmitter off
        ids  = range(nxt, nxt + K[m])                          # next packets from the queue
        nxt += K[m]
        bits = np.concatenate([packet_bits(i, p['battery']) for i in ids])
        rx   = demodulate(awgn(modulate(bits, MODES[m]), p['true_snr'][slot]), MODES[m])
        for j, i in enumerate(ids):                # ideal packet check: any bit error -> dropped
            if np.array_equal(rx[j*PKT_BITS:(j+1)*PKT_BITS], bits[j*PKT_BITS:(j+1)*PKT_BITS]):
                received.append(i)
                per_slot[slot] += 1
    return np.array(received, dtype=int), per_slot, used

def summarize(received):
    ttc   = 100 * np.sum(received < N_TTC) / N_TTC
    img   = 100 * np.sum((received >= N_TTC) & (received < N_TTC + N_IMG)) / N_IMG
    voice = np.sum(received >= N_TTC + N_IMG) * PKT_BITS / 3200 / 60   # minutes @ 3.2 kbps
    return ttc, img, voice, received.size * 128 / 1024                 # ..., total kB

def received_image(received):                                  # lost rows shown in red
    rgb = np.repeat(IMAGE[:, :, None], 3, axis=2).astype(float) / 255
    got = set(received.tolist())
    for r in range(N_IMG):
        if N_TTC + r not in got:
            rgb[r] = [0.85, 0.1, 0.1]
    return rgb

# =================================================================
# 6. RESULTS : 30 new test passes with random battery levels
# =================================================================
print("Testing on 30 new passes (real bits through the modem) ...")
test = [make_pass(rng.uniform(20, 100)) for _ in range(30)]
results = {name: np.array([summarize(run_downlink(p, f(p))[0]) for p in test])
           for name, f in STRATEGIES.items()}
agree = np.mean(np.concatenate([ai.predict(features(p, prio)) == best_mode(p['true_snr'], prio)
                                for p in test for prio in (1, 0)]))
print(f"  AI picks the ideal mode {100*agree:.1f}% of the time\n")
print(f"{'Strategy':15s}| TT&C  | Image | Voice (min) | Total (kB)")
for name, r in results.items():
    ttc, img, voice, kb = r.mean(axis=0)
    print(f"{name:15s}| {ttc:4.0f}% | {img:4.0f}% |    {voice:5.2f}    |  {kb:6.1f}")

# =================================================================
# 7. PLOTS
# =================================================================
full, low = make_pass(80), make_pass(30)                       # demo passes
rec_full, rate_full, modes_full = run_downlink(full, STRATEGIES['AI'](full))
prio_end = np.searchsorted(np.cumsum([K[m] for m in modes_full]), N_TTC + N_IMG)   # last priority slot

# Fig 1 - one pass: signal strength and the AI's decisions
fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
ax[0].plot(full['t'], full['true_snr'], lw=2, label='true downlink SNR')
ax[0].plot(full['t'], full['estimate'], '.', ms=3, alpha=0.5, label='AI input: link estimate')
ax[0].set(ylabel='SNR, Es/N0 (dB)', title='One pass at 80% battery and the AI\'s choices')
ax2 = ax[0].twinx()
ax2.plot(full['t'], full['elev'], 'k--', lw=1, label='elevation (reference only)')
ax2.set_ylabel('elevation (deg)')
ax[0].legend(loc='upper left')
ax2.legend(loc='upper right')
ax[1].axvspan(0, prio_end, color='C1', alpha=0.2, label='sending TT&C + image (play safe)')
ax[1].step(full['t'], modes_full, where='post', color='C3')
ax[1].legend(loc='lower center', fontsize=11)
ax[1].set(yticks=range(4), yticklabels=[NAME[m] for m in MODES], xlabel='time since pass start (s)', ylabel='AI mode')
for a in ax:
    a.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('fig1_pass_and_ai_decisions.png', dpi=150)

# Fig 2 - modem check: simulated BER matches theory; PER gives the switching points
snr = np.arange(0, 21, 2)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
for c, m in enumerate(MODES[1:]):
    sim = []
    for s in snr:
        b = rng.integers(0, 2, 300_000 // K[c+1] * K[c+1], dtype=np.uint8)
        sim.append(np.mean(demodulate(awgn(modulate(b, m), s), m) != b))
    fine = np.linspace(0, 20, 400)
    ax[0].semilogy(fine, ber_theory(m, fine), color=f'C{c}', label=f'{NAME[m]} theory')
    ax[0].semilogy(snr, np.maximum(sim, 1e-7), 'o', color=f'C{c}', label=f'{NAME[m]} simulated')
    ax[1].semilogy(fine, np.maximum(per(m, fine), 1e-7), color=f'C{c}', label=NAME[m])
ax[0].set(xlabel='SNR, Es/N0 (dB)', ylabel='bit error rate', ylim=(1e-6, 1), title='(a) BER: modem vs theory')
ax[1].axhline(TARGET_PER, color='k', ls='--', lw=1, label='1% target')
ax[1].set(xlabel='SNR, Es/N0 (dB)', ylabel='packet error rate (1024 bits)', ylim=(1e-4, 1.5),
          title='(b) Packet error rate (1024-bit packets)')
for a in ax:
    a.grid(alpha=0.3, which='both')
    a.legend(fontsize=10)
fig.suptitle('Modem check and switching points')
plt.tight_layout()
plt.savefig('fig2_ber_per.png', dpi=150)

# Fig 3 - data rate over the pass vs the Shannon limit
smooth = lambda v: np.convolve(v, np.ones(15), 'same') / np.convolve(np.ones(v.size), np.ones(15), 'same')
plt.figure(figsize=(10, 4.4))
plt.plot(full['t'], np.log2(1 + 10**(full['true_snr']/10)), 'k', lw=1.5, label='Shannon capacity')
for name in ['Fixed BPSK', 'Fixed 8-PSK', 'AI']:
    plt.plot(full['t'], smooth(run_downlink(full, STRATEGIES[name](full))[1]), lw=2, label=name)
plt.xlabel('time since pass start (s)')
plt.ylabel('correct bits per symbol')
plt.title('Data delivered during the pass (15 s average) vs Shannon limit')
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('fig3_rate_vs_shannon.png', dpi=150)

# Fig 4 - the image received on the ground
fig, ax = plt.subplots(2, 3, figsize=(11, 6.2))
for row, (p, names) in enumerate([(full, ['Fixed BPSK', 'Fixed 8-PSK', 'AI']),
                                  (low,  ['Fixed BPSK', 'Textbook rule', 'AI'])]):
    for col, name in enumerate(names):
        rec = run_downlink(p, STRATEGIES[name](p))[0]
        ttc, img, voice, kb = summarize(rec)
        ax[row, col].imshow(received_image(rec))
        ax[row, col].set_title(f'{name}, {p["battery"]:.0f}% battery\n'
                               f'TT&C {ttc:.0f}%  image {img:.0f}%  voice {voice:.1f} min', fontsize=12)
        ax[row, col].axis('off')
fig.suptitle('Image received after one pass (red rows = packets lost)')
plt.tight_layout(h_pad=2.0)
plt.savefig('fig4_received_images.png', dpi=150)

# Fig 5 - average over 30 test passes
names = list(results)
kb_ttc   = [results[n][:, 0].mean()/100 * N_TTC * 128/1024 for n in names]
kb_img   = [results[n][:, 1].mean()/100 * N_IMG * 128/1024 for n in names]
kb_voice = [results[n][:, 2].mean() * 60 * 3200 / 8 / 1024 for n in names]
plt.figure(figsize=(10, 4.6))
plt.bar(names, kb_ttc, label='TT&C', color='C3')
plt.bar(names, kb_img, bottom=kb_ttc, label='image', color='C2')
plt.bar(names, kb_voice, bottom=np.add(kb_ttc, kb_img), label='voice', color='C0')
for i, n in enumerate(names):
    plt.text(i, kb_ttc[i] + kb_img[i] + kb_voice[i] + 2, f'TT&C {results[n][:, 0].mean():.0f}%',
             ha='center', fontsize=11)
plt.ylabel('data delivered per pass (kB)')
plt.title('Average over 30 test passes (random battery levels)')
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('fig5_average_results.png', dpi=150)

# Fig 6 - what the AI learned: its mode at every link estimate
grid = np.arange(4, 26, 0.02)
rows = [(80, 1, 'Full power, priority data'), (80, 0, 'Full power, voice'),
        (30, 1, 'Low battery, priority data'), (30, 0, 'Low battery, voice')]
colours = ['#bbbbbb', 'C0', 'C1', 'C2']
fig, ax = plt.subplots(figsize=(10, 3.6))
for r, (batt, prio, label) in enumerate(rows):
    pol = ai.predict(np.column_stack([grid, np.full(grid.size, batt), np.full(grid.size, prio)]))
    for m in range(4):
        sel = pol == m
        ax.scatter(grid[sel], np.full(sel.sum(), r), marker='s', s=40, color=colours[m],
                   label=NAME[MODES[m]] if r == 0 or (r == 1 and m == 0) else None)
ax.set(yticks=range(4), yticklabels=[r[2] for r in rows], xlabel='link estimate (dB)',
       title='What the AI learned: its mode at every link estimate')
ax.invert_yaxis()
ax.grid(axis='x', alpha=0.3)
handles, labels = ax.get_legend_handles_labels()
order = [labels.index(NAME[m]) for m in MODES if NAME[m] in labels]
ax.legend([handles[i] for i in order], [labels[i] for i in order], ncol=4, loc='upper center',
          bbox_to_anchor=(0.5, -0.25), fontsize=12)
plt.tight_layout()
plt.savefig('fig6_ai_decision_map.png', dpi=150)
plt.show()

# =================================================================
# 8. LIVE DEMO : choose a battery level, fly a new pass
# =================================================================
def demo_pass(battery, n=5):
    passes = [make_pass(battery) for _ in range(n)]           # 5 passes: one pass is luck
    print(f"\n{n} new passes at {battery:.0f}% battery "
          f"({'LOW-POWER mode' if battery < 50 else 'full power'}), averages:")
    print(f"{'Strategy':15s}| TT&C  | Image | Voice (min) | Total (kB)")
    for name in ['Fixed BPSK', 'Fixed 8-PSK', 'Textbook rule', 'AI']:
        r = np.mean([summarize(run_downlink(p, STRATEGIES[name](p))[0]) for p in passes], axis=0)
        print(f"{name:15s}| {r[0]:4.0f}% | {r[1]:4.0f}% |    {r[2]:5.2f}    |  {r[3]:6.1f}")

try:
    while True:
        b = input("\nBattery level for the next pass, 20-100 (Enter to quit): ").strip()
        if not b:
            break
        try:
            demo_pass(min(max(float(b), 20), 100))
        except ValueError:
            print("Enter a number, e.g. 35")
except (EOFError, KeyboardInterrupt):
    pass
