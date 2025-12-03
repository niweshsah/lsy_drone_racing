import numpy as np
from scipy.optimize import minimize_scalar
from parallel_transport2 import GeometryEngine # Update filename if needed

class SpatialTransformer:
    def __init__(self, geo_engine):
        self.geo = geo_engine
        
    def get_frame_at_s(self, s_query):
        """Interpolates the PT frame at a specific s."""
        # Clamp s to valid range
        s_query = np.clip(s_query, 0, self.geo.total_length)
        
        # Find index (using precomputed s from geo engine)
        s_vals = self.geo.pt_frame['s']
        idx = np.searchsorted(s_vals, s_query) - 1
        idx = max(0, min(idx, len(s_vals) - 2))
        
        # Simple Linear Interpolation for stability
        ds = s_vals[idx+1] - s_vals[idx]
        alpha = (s_query - s_vals[idx]) / ds
        
        # Interpolate Basis Vectors
        # Note: Linear interp of vectors de-normalizes them slightly, 
        # but for small steps it's negligible. Normalizing ensures safety.
        t = (1-alpha)*self.geo.pt_frame['t'][idx] + alpha*self.geo.pt_frame['t'][idx+1]
        n1 = (1-alpha)*self.geo.pt_frame['n1'][idx] + alpha*self.geo.pt_frame['n1'][idx+1]
        n2 = (1-alpha)*self.geo.pt_frame['n2'][idx] + alpha*self.geo.pt_frame['n2'][idx+1]
        
        # Interpolate Curvature
        k1 = (1-alpha)*self.geo.pt_frame['k1'][idx] + alpha*self.geo.pt_frame['k1'][idx+1]
        k2 = (1-alpha)*self.geo.pt_frame['k2'][idx] + alpha*self.geo.pt_frame['k2'][idx+1]
        
        # Renormalize
        t /= np.linalg.norm(t)
        n1 /= np.linalg.norm(n1)
        n2 /= np.linalg.norm(n2)
        
        return {'t': t, 'n1': n1, 'n2': n2, 'k1': k1, 'k2': k2}

    def cartesian_to_spatial(self, pos_inertial, vel_inertial, s_guess):
        """
        Projects (pos, vel) -> (s, w1, w2, ds, dw1, dw2).
        Algorithm:
        1. Find closest point on spline (s).
        2. Project deviation vector onto n1, n2 (w1, w2).
        3. Rotate velocity into frame and scale longitudinal component (ds).
        """
        # 1. Find s (Projection)
        # Step A: Coarse Search (avoid local minima)
        search_radius = 2.0 # meters
        s_samples = np.linspace(s_guess - search_radius, s_guess + search_radius, 50)
        s_samples = np.clip(s_samples, 0, self.geo.total_length)
        
        # Compute distances to spline
        points_on_path = self.geo.spline(s_samples)
        dists = np.linalg.norm(points_on_path - pos_inertial, axis=1)
        s_coarse = s_samples[np.argmin(dists)]
        
        # Step B: Refine s using optimization to ensure orthogonality
        # This is critical for accurate velocity decomposition
        bnds = (max(0, s_coarse - 0.2), min(self.geo.total_length, s_coarse + 0.2))
        
        res = minimize_scalar(
            lambda s: np.linalg.norm(self.geo.spline(s) - pos_inertial),
            bounds=bnds,
            method='bounded',
            options={'xatol': 1e-6}
        )
        s_exact = res.x
        
        # Get Frame at projected s
        frame = self.get_frame_at_s(s_exact)
        p_path = self.geo.spline(s_exact)
        
        # 2. Position Transformation (w1, w2)
        err_vec = pos_inertial - p_path
        w1 = np.dot(err_vec, frame['n1'])
        w2 = np.dot(err_vec, frame['n2'])
        
        # 3. Velocity Transformation
        # v_inertial = R * v_local
        # v_local = R.T * v_inertial = [v_t, v_n1, v_n2]
        R_mat = np.column_stack((frame['t'], frame['n1'], frame['n2']))
        v_local = R_mat.T @ vel_inertial
        
        # Scaling factor h = 1 - k1*w1 - k2*w2
        h = 1.0 - frame['k1']*w1 - frame['k2']*w2
        
        # Handle singularity if drone is excessively far from path center
        if h < 0.1: h = 0.1 
        
        ds = v_local[0] / h
        dw1 = v_local[1]
        dw2 = v_local[2]
        
        return np.array([s_exact, w1, w2, ds, dw1, dw2])

    def spatial_to_cartesian(self, spatial_state):
        """
        Reconstructs (pos, vel) from spatial state.
        Used for verification.
        """
        s, w1, w2, ds, dw1, dw2 = spatial_state
        
        frame = self.get_frame_at_s(s)
        p_path = self.geo.spline(s)
        
        # Pos = Path + w1*n1 + w2*n2
        pos_inertial = p_path + w1*frame['n1'] + w2*frame['n2']
        
        # Vel = R * [ds*h, dw1, dw2]
        h = 1.0 - frame['k1']*w1 - frame['k2']*w2
        v_local = np.array([ds * h, dw1, dw2])
        
        R_mat = np.column_stack((frame['t'], frame['n1'], frame['n2']))
        vel_inertial = R_mat @ v_local
        
        return pos_inertial, vel_inertial

def run_round_trip_test():
    print("\n--- Step 2: Coordinate Transformer Verification ---")
    
    # Init Engine
    geo = GeometryEngine()
    transformer = SpatialTransformer(geo)
    
    # Test Case: Drone is slightly off-path and moving diagonally
    # Pick a point on the path at s=5.0
    s_target = 5.0
    frame = transformer.get_frame_at_s(s_target)
    
    # Create a "Phantom Drone" at (s=5, w1=0.5, w2=-0.2)
    # Moving with ds=2.0, dw1=0.1, dw2=0.0
    spatial_ground_truth = np.array([5.0, 0.5, -0.2, 2.0, 0.1, 0.0])
    
    print(f"Ground Truth Spatial: {spatial_ground_truth}")
    
    # 1. Convert Spatial -> Cartesian (Manual math logic check)
    pos_inertial, vel_inertial = transformer.spatial_to_cartesian(spatial_ground_truth)
    print(f"Cartesian Pos: {pos_inertial}")
    
    # 2. Convert Cartesian -> Spatial (The Algorithm under test)
    # Pass a guess close to 5.0
    spatial_recovered = transformer.cartesian_to_spatial(pos_inertial, vel_inertial, s_guess=4.5)
    print(f"Recovered Spatial:  {spatial_recovered}")
    
    # 3. Validation
    error = np.linalg.norm(spatial_ground_truth - spatial_recovered)
    print(f"Round Trip Error: {error:.2e}")
    
    if error < 1e-4:
        print(">> SUCCESS: Coordinate Transform is robust.")
    else:
        print(">> FAILURE: Transform math is incorrect.")

if __name__ == "__main__":
    run_round_trip_test()