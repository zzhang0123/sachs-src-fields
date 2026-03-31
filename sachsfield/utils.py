"""
Visualization and summary statistics utilities.

Provides:
  - animate_evolution: mollview / imshow animation over lambda steps
  - pixel_pdf: 1D PDF (histogram) of pixel values
  - angular_power_spectrum: C_ell measurement for full-sky or flat-sky results
  - summary_statistics: combined PDF + power spectrum panel figure
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# ---------------------------------------------------------------------------
# Animation
# ---------------------------------------------------------------------------

def animate_evolution(result, mode='fullsky', nside=None, nx=None, ny=None, dx=None,
                      fields=None, interval=200, vrange=None,
                      show_saddle=True, title_prefix=''):
    """Animate the evolved field maps over lambda steps.

    Parameters
    ----------
    result : FullResult
        Output from SachsFieldSolver.solve().
    mode : str
        'fullsky' for healpy mollview, 'flatsky' for imshow.
    nside : int
        Required if mode='fullsky'.
    nx, ny : int
        Required if mode='flatsky'.
    dx : float
        Pixel size in radians (for flatsky axis labels).
    fields : list of int, optional
        Which field indices to show. Default: all fields.
    interval : int
        Milliseconds between frames.
    vrange : dict, optional
        {field_index: (vmin, vmax)} to fix color range. If None, auto-scales
        per frame.
    show_saddle : bool
        If True, show total field (chi + xi). If False, show xi only.
    title_prefix : str
        Prepended to each subplot title.

    Returns
    -------
    anim : FuncAnimation
        Matplotlib animation object. Call plt.show() or anim.save() to use.
    """
    n_fields = result.fluctuation.n_fields
    n_times = len(result.fluctuation.lam)
    if fields is None:
        fields = list(range(n_fields))
    n_panels = len(fields)

    field_labels = {0: r'$x_1$', 1: r'$x_2$', 2: r'$x_3$', 3: r'$x_4$'}

    if mode == 'fullsky':
        import healpy as hp
        fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 4),
                                 subplot_kw={'projection': 'mollweide'})
        if n_panels == 1:
            axes = [axes]

        # Pre-compute all frames
        all_maps = []
        for t_idx in range(n_times):
            if show_saddle:
                maps = result.total_field(t_idx)
            else:
                maps = result.fluctuation.xi[t_idx]
            all_maps.append(maps)

        images = []
        for i, a in enumerate(fields):
            m = all_maps[0][a]
            # Convert healpix map to mollweide projection
            theta, phi = hp.pix2ang(nside, np.arange(len(m)))
            lon = phi - np.pi
            lat = np.pi / 2 - theta
            sc = axes[i].scatter(lon, lat, c=m, s=0.1, cmap='RdBu_r', rasterized=True)
            images.append(sc)
            axes[i].set_title(f'{title_prefix}{field_labels.get(a, f"$x_{a+1}$")}')

        def update(t_idx):
            for i, a in enumerate(fields):
                m = all_maps[t_idx][a]
                images[i].set_array(m)
                if vrange and a in vrange:
                    images[i].set_clim(*vrange[a])
                else:
                    images[i].set_clim(m.min(), m.max())
                lam_val = result.fluctuation.lam[t_idx]
                axes[i].set_title(
                    f'{title_prefix}{field_labels.get(a, f"$x_{a+1}$")} '
                    f'$\\lambda={lam_val:.3f}$')
            return images

    elif mode == 'flatsky':
        fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4))
        if n_panels == 1:
            axes = [axes]

        extent = None
        if dx is not None and nx is not None and ny is not None:
            extent = [0, np.degrees(nx * dx), 0, np.degrees(ny * dx)]

        images = []
        for i, a in enumerate(fields):
            if show_saddle:
                m = result.total_field(0)[a].reshape(nx, ny)
            else:
                m = result.fluctuation.xi[0, a].reshape(nx, ny)
            im = axes[i].imshow(m, origin='lower', extent=extent, cmap='RdBu_r',
                                aspect='equal')
            images.append(im)
            axes[i].set_xlabel('deg' if dx else 'pixel')
            axes[i].set_ylabel('deg' if dx else 'pixel')
            plt.colorbar(im, ax=axes[i], shrink=0.8)

        def update(t_idx):
            for i, a in enumerate(fields):
                if show_saddle:
                    m = result.total_field(t_idx)[a].reshape(nx, ny)
                else:
                    m = result.fluctuation.xi[t_idx, a].reshape(nx, ny)
                images[i].set_data(m)
                if vrange and a in vrange:
                    images[i].set_clim(*vrange[a])
                else:
                    images[i].set_clim(m.min(), m.max())
                lam_val = result.fluctuation.lam[t_idx]
                axes[i].set_title(
                    f'{title_prefix}{field_labels.get(a, f"$x_{a+1}$")} '
                    f'$\\lambda={lam_val:.3f}$')
            return images
    else:
        raise ValueError(f"mode must be 'fullsky' or 'flatsky', got '{mode}'")

    fig.tight_layout()
    anim = FuncAnimation(fig, update, frames=n_times, interval=interval, blit=False)
    return anim


# ---------------------------------------------------------------------------
# 1D Pixel PDF
# ---------------------------------------------------------------------------

def pixel_pdf(maps, n_bins=50, fields=None, field_labels=None,
              log_counts=False, ax=None):
    """Compute and plot the 1D PDF (histogram) of pixel values.

    Parameters
    ----------
    maps : ndarray, shape (n_fields, npix)
        Field maps at a single lambda step (e.g., from result.total_field(idx)
        or result.fluctuation.xi[idx]).
    n_bins : int
        Number of histogram bins.
    fields : list of int, optional
        Which field indices to plot. Default: all.
    field_labels : list of str, optional
        Custom labels. Default: x_1, x_2, ...
    log_counts : bool
        If True, use log scale on y-axis.
    ax : matplotlib Axes, optional
        If provided, plot on this axes.

    Returns
    -------
    hist_data : list of (bin_centers, counts, bin_edges)
        One tuple per field.
    ax : matplotlib Axes
    """
    n_fields = maps.shape[0]
    if fields is None:
        fields = list(range(n_fields))
    if field_labels is None:
        field_labels = [f'$x_{a+1}$' for a in fields]

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))

    hist_data = []
    for i, a in enumerate(fields):
        counts, bin_edges = np.histogram(maps[a], bins=n_bins, density=True)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        ax.step(bin_centers, counts, where='mid', label=field_labels[i])
        hist_data.append((bin_centers, counts, bin_edges))

    ax.set_xlabel('Pixel value')
    ax.set_ylabel('PDF')
    if log_counts:
        ax.set_yscale('log')
    ax.legend()
    return hist_data, ax


# ---------------------------------------------------------------------------
# Angular Power Spectrum
# ---------------------------------------------------------------------------

def angular_power_spectrum(maps, mode='fullsky', nside=None,
                           nx=None, ny=None, dx=None,
                           fields=None, lmax=None):
    """Measure the angular (auto) power spectrum of field maps.

    Parameters
    ----------
    maps : ndarray, shape (n_fields, npix)
        Field maps at a single lambda step.
    mode : str
        'fullsky' uses healpy.anafast, 'flatsky' uses 2D FFT.
    nside : int
        Required for fullsky.
    nx, ny : int
        Required for flatsky.
    dx : float
        Pixel size in radians, required for flatsky.
    fields : list of int, optional
        Which fields. Default: all.
    lmax : int, optional
        Maximum ell for fullsky. Default: 3*nside - 1.

    Returns
    -------
    ells : ndarray
        Multipole values (fullsky) or bin centers (flatsky).
    cls : dict
        {field_index: C_ell array}. For fullsky, auto-spectra only.
        For flatsky, binned 2D power spectrum.
    """
    n_fields = maps.shape[0]
    if fields is None:
        fields = list(range(n_fields))

    cls = {}

    if mode == 'fullsky':
        import healpy as hp
        if lmax is None:
            lmax = 3 * nside - 1
        ells = np.arange(lmax + 1)
        for a in fields:
            cls[a] = hp.anafast(maps[a], lmax=lmax)

    elif mode == 'flatsky':
        kx = np.fft.fftfreq(nx, d=dx) * 2.0 * np.pi
        ky = np.fft.fftfreq(ny, d=dx) * 2.0 * np.pi
        kx2d, ky2d = np.meshgrid(kx, ky, indexing='ij')
        ell_grid = np.sqrt(kx2d**2 + ky2d**2)

        ell_max = np.max(ell_grid) / 2.0
        n_bins = min(40, nx // 2)
        ell_edges = np.linspace(0, ell_max, n_bins + 1)
        ells = 0.5 * (ell_edges[:-1] + ell_edges[1:])

        for a in fields:
            map_2d = maps[a].reshape(nx, ny)
            fk = np.fft.fft2(map_2d) * dx**2
            pk = np.abs(fk)**2 / (nx * ny * dx**2)
            cl_binned = np.zeros(n_bins)
            for b in range(n_bins):
                mask = (ell_grid >= ell_edges[b]) & (ell_grid < ell_edges[b + 1])
                if np.any(mask):
                    cl_binned[b] = np.mean(pk[mask])
            cls[a] = cl_binned
    else:
        raise ValueError(f"mode must be 'fullsky' or 'flatsky', got '{mode}'")

    return ells, cls


def plot_power_spectra(ells, cls, field_labels=None, ax=None, **plot_kwargs):
    """Plot angular power spectra.

    Parameters
    ----------
    ells : ndarray
    cls : dict {field_index: C_ell}
    field_labels : dict {field_index: str}, optional
    ax : matplotlib Axes, optional
    **plot_kwargs : passed to ax.loglog

    Returns
    -------
    ax : matplotlib Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))
    if field_labels is None:
        field_labels = {a: f'$C_\\ell^{{{a+1}{a+1}}}$' for a in cls}

    for a, cl in cls.items():
        mask = (ells > 0) & (cl > 0)
        ax.loglog(ells[mask], cl[mask], label=field_labels.get(a, f'field {a}'),
                  **plot_kwargs)

    ax.set_xlabel(r'$\ell$')
    ax.set_ylabel(r'$C_\ell$')
    ax.legend()
    return ax


# ---------------------------------------------------------------------------
# Combined summary statistics
# ---------------------------------------------------------------------------

def summary_statistics(result, lam_index=-1, mode='fullsky',
                       nside=None, nx=None, ny=None, dx=None,
                       fields=None, show_saddle=True, n_bins=50,
                       suptitle=None):
    """Plot a 3-panel summary: map, PDF, and power spectrum.

    Parameters
    ----------
    result : FullResult
        Output from SachsFieldSolver.solve().
    lam_index : int
        Which output time step. Default: last.
    mode : str
        'fullsky' or 'flatsky'.
    nside : int
        For fullsky.
    nx, ny : int
        For flatsky.
    dx : float
        For flatsky.
    fields : list of int, optional
        Which fields. Default: all.
    show_saddle : bool
        If True, show total field (chi + xi). If False, xi only.
    n_bins : int
        Histogram bins for PDF.
    suptitle : str, optional
        Figure title.

    Returns
    -------
    fig : matplotlib Figure
    stats : dict
        'pdf': list of (bin_centers, counts, bin_edges) per field,
        'ells': ell array,
        'cls': {field_index: C_ell},
        'mean': per-field mean,
        'std': per-field std,
        'skew': per-field skewness,
        'kurtosis': per-field excess kurtosis.
    """
    from scipy.stats import skew, kurtosis

    n_fields = result.fluctuation.n_fields
    if fields is None:
        fields = list(range(n_fields))

    if show_saddle:
        maps = result.total_field(lam_index)
    else:
        maps = result.fluctuation.xi[lam_index]

    lam_val = result.fluctuation.lam[lam_index]
    field_labels_map = {a: f'$x_{a+1}$' for a in fields}
    field_labels_list = [field_labels_map[a] for a in fields]

    # --- Layout: maps (top row), PDF (bottom left), Cl (bottom right) ---
    n_panels = len(fields)
    fig = plt.figure(figsize=(max(12, 4 * n_panels), 9))

    # Top row: maps
    if mode == 'fullsky':
        import healpy as hp
        for i, a in enumerate(fields):
            ax_map = fig.add_subplot(2, n_panels, i + 1)
            # Use a simple 2D projection for inline display
            m = maps[a]
            theta, phi = hp.pix2ang(nside, np.arange(len(m)))
            lon_deg = np.degrees(phi)
            lat_deg = 90.0 - np.degrees(theta)
            sc = ax_map.scatter(lon_deg, lat_deg, c=m, s=0.3,
                                cmap='RdBu_r', rasterized=True)
            ax_map.set_xlabel('lon [deg]')
            ax_map.set_ylabel('lat [deg]')
            ax_map.set_title(f'{field_labels_map[a]}  $\\lambda={lam_val:.3f}$')
            plt.colorbar(sc, ax=ax_map, shrink=0.7)

    elif mode == 'flatsky':
        extent = None
        if dx is not None:
            extent = [0, np.degrees(nx * dx), 0, np.degrees(ny * dx)]
        for i, a in enumerate(fields):
            ax_map = fig.add_subplot(2, n_panels, i + 1)
            m2d = maps[a].reshape(nx, ny)
            im = ax_map.imshow(m2d, origin='lower', extent=extent,
                               cmap='RdBu_r', aspect='equal')
            ax_map.set_xlabel('deg' if dx else 'pixel')
            ax_map.set_ylabel('deg' if dx else 'pixel')
            ax_map.set_title(f'{field_labels_map[a]}  $\\lambda={lam_val:.3f}$')
            plt.colorbar(im, ax=ax_map, shrink=0.7)

    # Bottom left: PDF
    ax_pdf = fig.add_subplot(2, 2, 3)
    pdf_data, _ = pixel_pdf(maps, n_bins=n_bins, fields=fields,
                            field_labels=field_labels_list, ax=ax_pdf)
    ax_pdf.set_title('Pixel PDF')

    # Bottom right: Power spectrum
    ax_cl = fig.add_subplot(2, 2, 4)
    ells, cls = angular_power_spectrum(
        maps, mode=mode, nside=nside, nx=nx, ny=ny, dx=dx, fields=fields)
    cl_labels = {a: f'$C_\\ell^{{{a+1}{a+1}}}$' for a in fields}
    plot_power_spectra(ells, cls, field_labels=cl_labels, ax=ax_cl)
    ax_cl.set_title('Angular Power Spectrum')

    if suptitle:
        fig.suptitle(suptitle, fontsize=14, y=1.02)
    try:
        fig.tight_layout()
    except Exception:
        pass  # skip if subplot grid is incompatible

    # Compute scalar statistics
    stats = {
        'pdf': pdf_data,
        'ells': ells,
        'cls': cls,
        'mean': {a: float(np.mean(maps[a])) for a in fields},
        'std': {a: float(np.std(maps[a])) for a in fields},
        'skew': {a: float(skew(maps[a])) for a in fields},
        'kurtosis': {a: float(kurtosis(maps[a])) for a in fields},
    }

    return fig, stats


def evolution_summary(result, mode='fullsky', nside=None,
                      nx=None, ny=None, dx=None,
                      fields=None, show_saddle=False):
    """Plot evolution of RMS, skewness, and kurtosis over lambda.

    Parameters
    ----------
    result : FullResult
    mode, nside, nx, ny, dx : same as summary_statistics
    fields : list of int, optional
    show_saddle : bool
        If True, statistics of total field. If False, xi only.

    Returns
    -------
    fig : matplotlib Figure
    time_stats : dict
        'lam': lambda array,
        'rms': {field: array}, 'skew': {field: array}, 'kurtosis': {field: array}
    """
    from scipy.stats import skew, kurtosis

    n_fields = result.fluctuation.n_fields
    n_times = len(result.fluctuation.lam)
    if fields is None:
        fields = list(range(n_fields))

    lam_arr = result.fluctuation.lam
    rms = {a: np.zeros(n_times) for a in fields}
    skew_arr = {a: np.zeros(n_times) for a in fields}
    kurt_arr = {a: np.zeros(n_times) for a in fields}

    for t in range(n_times):
        if show_saddle:
            maps = result.total_field(t)
        else:
            maps = result.fluctuation.xi[t]
        for a in fields:
            rms[a][t] = np.std(maps[a])
            skew_arr[a][t] = skew(maps[a])
            kurt_arr[a][t] = kurtosis(maps[a])

    field_labels = {a: f'$x_{a+1}$' for a in fields}

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for a in fields:
        axes[0].plot(lam_arr, rms[a], label=field_labels[a])
    axes[0].set_xlabel(r'$\lambda$')
    axes[0].set_ylabel('RMS')
    axes[0].set_title('RMS evolution')
    axes[0].legend()

    for a in fields:
        axes[1].plot(lam_arr, skew_arr[a], label=field_labels[a])
    axes[1].set_xlabel(r'$\lambda$')
    axes[1].set_ylabel('Skewness')
    axes[1].set_title('Skewness evolution')
    axes[1].axhline(0, color='k', ls=':', alpha=0.3)
    axes[1].legend()

    for a in fields:
        axes[2].plot(lam_arr, kurt_arr[a], label=field_labels[a])
    axes[2].set_xlabel(r'$\lambda$')
    axes[2].set_ylabel('Excess kurtosis')
    axes[2].set_title('Kurtosis evolution')
    axes[2].axhline(0, color='k', ls=':', alpha=0.3)
    axes[2].legend()

    fig.tight_layout()

    time_stats = {
        'lam': lam_arr,
        'rms': rms,
        'skew': skew_arr,
        'kurtosis': kurt_arr,
    }
    return fig, time_stats
