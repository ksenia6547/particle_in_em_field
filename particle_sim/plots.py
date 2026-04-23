import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import glob
import os

def list_csv_files():
    return glob.glob('output/*.csv')

def plot_trajectories(filename, output_dir='plots'):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(filename, skipinitialspace=True, encoding='utf-8-sig')
    
    fig = plt.figure(figsize=(14, 10))
    
    #3d траектория
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.plot(df['x'], df['y'], df['z'], 'b-', linewidth=0.7, alpha=0.7)
    ax1.scatter(df['x'].iloc[0], df['y'].iloc[0], df['z'].iloc[0], c='g', s=50, label='Start')
    ax1.scatter(df['x'].iloc[-1], df['y'].iloc[-1], df['z'].iloc[-1], c='r', s=50, label='End')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('3D Trajectory')
    ax1.legend(loc='upper right')
    ax1.set_box_aspect([1,1,1])
    
    #xy проекция
    ax2 = fig.add_subplot(222)
    ax2.plot(df['x'], df['y'], 'b-', linewidth=0.7)
    ax2.scatter(df['x'].iloc[0], df['y'].iloc[0], c='g', s=50, label='Start')
    ax2.scatter(df['x'].iloc[-1], df['y'].iloc[-1], c='r', s=50, label='End')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('XY Projection')
    ax2.set_aspect('equal')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    #xz проекция
    ax3 = fig.add_subplot(223)
    ax3.plot(df['x'], df['z'], 'b-', linewidth=0.7)
    ax3.scatter(df['x'].iloc[0], df['z'].iloc[0], c='g', s=50, label='Start')
    ax3.scatter(df['x'].iloc[-1], df['z'].iloc[-1], c='r', s=50, label='End')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Z')
    ax3.set_title('XZ Projection')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    #yz проекция
    ax4 = fig.add_subplot(224)
    ax4.plot(df['y'], df['z'], 'b-', linewidth=0.7)
    ax4.scatter(df['y'].iloc[0], df['z'].iloc[0], c='g', s=50, label='Start')
    ax4.scatter(df['y'].iloc[-1], df['z'].iloc[-1], c='r', s=50, label='End')
    ax4.set_xlabel('Y')
    ax4.set_ylabel('Z')
    ax4.set_title('YZ Projection')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle(os.path.basename(filename).replace('.csv', ''), fontsize=14)
    plt.tight_layout()
    outname = os.path.join(output_dir, os.path.basename(filename).replace('.csv', '_trajectories.png'))
    plt.savefig(outname, dpi=150)
    plt.close()
    print(f"  Saved: {outname}")

def plot_energy_and_phase(filename, output_dir='plots'):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(filename, skipinitialspace=True, encoding='utf-8-sig')
    
    df['E_kin'] = 0.5 * (df['vx']**2 + df['vy']**2 + df['vz']**2)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    #кинетическая энергия
    ax1 = axes[0, 0]
    ax1.plot(df['t'], df['E_kin'], 'b-', linewidth=0.7)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Kinetic Energy')
    ax1.set_title('Kinetic Energy vs Time')
    ax1.grid(True, alpha=0.3)
    
    #фазовый портрет vx, x
    ax2 = axes[0, 1]
    ax2.plot(df['x'], df['vx'], 'b-', linewidth=0.5, alpha=0.7)
    ax2.scatter(df['x'].iloc[0], df['vx'].iloc[0], c='g', s=50, label='Start')
    ax2.scatter(df['x'].iloc[-1], df['vx'].iloc[-1], c='r', s=50, label='End')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Vx')
    ax2.set_title('Phase Portrait (Vx, X)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    #фазовый портрет vy, y    
    ax3 = axes[1, 0]
    ax3.plot(df['y'], df['vy'], 'b-', linewidth=0.5, alpha=0.7)
    ax3.scatter(df['y'].iloc[0], df['vy'].iloc[0], c='g', s=50, label='Start')
    ax3.scatter(df['y'].iloc[-1], df['vy'].iloc[-1], c='r', s=50, label='End')
    ax3.set_xlabel('Y')
    ax3.set_ylabel('Vy')
    ax3.set_title('Phase Portrait (Vy, Y)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    #фазовый портрет vy, y    
    ax4 = axes[1, 1]
    ax4.plot(df['z'], df['vz'], 'b-', linewidth=0.5, alpha=0.7)
    ax4.scatter(df['z'].iloc[0], df['vz'].iloc[0], c='g', s=50, label='Start')
    ax4.scatter(df['z'].iloc[-1], df['vz'].iloc[-1], c='r', s=50, label='End')
    ax4.set_xlabel('Z')
    ax4.set_ylabel('Vz')
    ax4.set_title('Phase Portrait (Vz, Z)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle(os.path.basename(filename).replace('.csv', ''), fontsize=14)
    plt.tight_layout()
    outname = os.path.join(output_dir, os.path.basename(filename).replace('.csv', '_energy_phase.png'))
    plt.savefig(outname, dpi=150)
    plt.close()
    print(f"  Saved: {outname}")

def plot_radius(filename, output_dir='plots'):

    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(filename, skipinitialspace=True, encoding='utf-8-sig')
    r = np.sqrt(df['x']**2 + df['y']**2)

    #ларморовский радиус
    plt.figure(figsize=(10, 6))
    plt.plot(df['t'], r, 'b-', linewidth=0.7)
    plt.xlabel('Time')
    plt.ylabel(r'Larmor Radius $r_L = \sqrt{x^2+y^2}$')
    plt.title(os.path.basename(filename).replace('.csv', '') + ' (Larmor Radius)')
    plt.grid(True, alpha=0.3)
    outname = os.path.join(output_dir, os.path.basename(filename).replace('.csv', '_radius.png'))
    plt.savefig(outname, dpi=150)
    plt.close()
    print(f"  Saved: {outname}")


def _interpolate_to_ref(df_ref, df_other):

    t_ref   = df_ref['t'].values
    t_other = df_other['t'].values
    t_min   = max(t_ref[0],  t_other[0])
    t_max   = min(t_ref[-1], t_other[-1])
    mask    = (t_ref >= t_min) & (t_ref <= t_max)
    t_common = t_ref[mask]

    interp = pd.DataFrame({'t': t_common})
    for col in ['x', 'y', 'z', 'vx', 'vy', 'vz']:
        interp[col] = np.interp(t_common, t_other, df_other[col].values)

    return interp, df_ref[mask].reset_index(drop=True)

def compare_methods():
    ref_path = 'output/dop853.csv'
    if not os.path.exists(ref_path):
        print("\n  ERROR: Reference file 'output/dop853.csv' not found.")
        print("  Please run the simulation first to generate dop853.csv.")
        return

    df_ref = pd.read_csv(ref_path, skipinitialspace=True, encoding='utf-8-sig')
    t_end  = df_ref['t'].iloc[-1]
    ref_end = df_ref[['x', 'y', 'z', 'vx', 'vy', 'vz']].iloc[-1].values

    files = sorted(
        f for f in list_csv_files()
        if os.path.basename(f) != 'dop853.csv'
    )

    if not files:
        print("\n  No other CSV files found to compare against dop853.")
        return

    W0, W1, W2 = 12, 9, 14

    header = (
        f"{'Method':<{W0}}"
        f"{'Steps':>{W1}}"
        f"{'dt_mean':>{W2}}"
        f"{'pos_err_end':>{W2}}"
        f"{'vel_err_end':>{W2}}"
        f"{'pos_err_max':>{W2}}"
        f"{'pos_err_rms':>{W2}}"
        f"{'vel_err_rms':>{W2}}"
        f"{'E_drift_%':>{W2}}"
    )
    sep = '-' * len(header)

    print()
    print('=' * len(header))
    print('  Comparison  ')
    print(f'  Reference endpoint: t = {t_end:.4f}')
    print('=' * len(header))
    print(header)
    print(sep)

    for fpath in files:
        name = os.path.basename(fpath).replace('.csv', '')
        try:
            df = pd.read_csv(fpath, skipinitialspace=True, encoding='utf-8-sig')
        except Exception as e:
            print(f"  Cannot read {fpath}: {e}")
            continue

        required = ['t', 'x', 'y', 'z', 'vx', 'vy', 'vz']
        if not all(c in df.columns for c in required):
            print(f"{name:<{W0}}  missing columns, skipped")
            continue

        steps   = len(df) - 1
        dt_mean = (df['t'].iloc[-1] - df['t'].iloc[0]) / steps if steps > 0 else float('nan')

        idx_end = (df['t'] - t_end).abs().idxmin()
        row_end = df[['x', 'y', 'z', 'vx', 'vy', 'vz']].iloc[idx_end].values
        pos_err_end = np.linalg.norm(row_end[:3]  - ref_end[:3])
        vel_err_end = np.linalg.norm(row_end[3:6] - ref_end[3:6])

        df_i, df_r = _interpolate_to_ref(df_ref, df)

        d_pos = np.sqrt(
            (df_i['x'].values  - df_r['x'].values)**2 +
            (df_i['y'].values  - df_r['y'].values)**2 +
            (df_i['z'].values  - df_r['z'].values)**2
        )
        d_vel = np.sqrt(
            (df_i['vx'].values - df_r['vx'].values)**2 +
            (df_i['vy'].values - df_r['vy'].values)**2 +
            (df_i['vz'].values - df_r['vz'].values)**2
        )

        pos_err_max = float(np.max(d_pos))
        pos_err_rms = float(np.sqrt(np.mean(d_pos**2)))
        vel_err_rms = float(np.sqrt(np.mean(d_vel**2)))

        E0 = 0.5 * (df['vx'].iloc[0]**2  + df['vy'].iloc[0]**2  + df['vz'].iloc[0]**2)
        E1 = 0.5 * (df['vx'].iloc[-1]**2 + df['vy'].iloc[-1]**2 + df['vz'].iloc[-1]**2)
        e_drift = abs(E1 - E0) / (E0 + 1e-300) * 100.0

        print(
            f"{name:<{W0}}"
            f"{steps:>{W1}d}"
            f"{dt_mean:>{W2}.4e}"
            f"{pos_err_end:>{W2}.4e}"
            f"{vel_err_end:>{W2}.4e}"
            f"{pos_err_max:>{W2}.4e}"
            f"{pos_err_rms:>{W2}.4e}"
            f"{vel_err_rms:>{W2}.4e}"
            f"{e_drift:>{W2}.4e}"
        )

    print(sep)
    print()

def main():
    files = list_csv_files()
    if not files:
        print("Error: No CSV files found in 'output/' directory.")
        print("Please run the simulation first (./particle_sim).")
        return
    
    while True:
        print("Available CSV files:")
        for i, f in enumerate(files):
            print(f"{i+1}. {os.path.basename(f)}")
        print(f"{len(files)+2}. Comparison")
        print("0. Exit")
        
        choice = input("\nSelect file number: ")
        
        if choice == "0":
            print("Exiting.")
            break
        
        try:
            idx = int(choice) - 1
                
            if idx == len(files) + 1:
                compare_methods()
                
            elif 0 <= idx < len(files):
                file = files[idx]
                print(f"\nGenerating plots for: {os.path.basename(file)}")
                plot_trajectories(file)
                plot_energy_and_phase(file)
                plot_radius(file)
              
            else:
                print("Invalid selection. Please try again.")
                
        except ValueError:
            print("Invalid input. Please enter a number.")
            continue

if __name__ == '__main__':
    main()
