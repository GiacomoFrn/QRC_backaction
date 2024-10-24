import numpy as np

class HilbertSpace:
    """
    This class is used to generate the basis for the Hilbert space
    """

    def __init__(self, n, system, flag_operators=True):
        self.n = n
        self.system = system
        self.basis = self.generate_basis()
        # generate local pauli operators
        self.id_local = np.eye(2)
        self.sigma_x = np.array([[0, 1.], [1., 0]])
        self.sigma_y = 1j*np.array([[0, -1.], [1., 0]])
        self.sigma_z = np.array([[1., 0], [0, -1.]])
        if flag_operators:
            if self.system == 'spin':
                self.generate_spin_operators()
            elif self.system == 'photon':
                self.generate_photon_operators()

    def generate_basis(self):
        # Implement the logic to generate basis based on the system type
        if self.system == 'photon':
            return self.generate_photon_fock_basis()
        elif self.system == 'spin':
            return self.generate_spin_fock_basis()

    def generate_photon_fock_basis(self):
        # Generate the basis for the photon fock space
        # n the number of modes, nmodes + 1 is dim of the space
        # i.e. vacuum mode + nmodes
        self.dimension = self.n + 1
        fock_basis = []
        for i in range(self.dimension):
            v_ph = np.zeros((self.dimension), dtype=int)
            v_ph[i] = 1
            fock_basis.append(tuple(v_ph))
        return fock_basis

    def generate_spin_fock_basis(self):
        # Generate the basis for the spin fock space
        self.dimension = 2 ** self.n
        spin_fock_basis = []
        for i in range(self.dimension):
            binary_repr = np.binary_repr(i, width=self.n)
            spin_fock_basis.append(tuple([int(bit) for bit in binary_repr]))
        spin_fock_basis.reverse()
        return spin_fock_basis

    def generate_spin_operators(self):
        # Generate the spin operators for the spin fock space
        self.X = {}
        self.Y = {}
        self.Z = {}
        for i in range(1, self.n+1):
            self.X[i] = np.kron(
                np.eye(2**(i-1)), np.kron(self.sigma_x, np.eye(2**(self.n-i))))
            self.Y[i] = np.kron(
                np.eye(2**(i-1)), np.kron(self.sigma_y, np.eye(2**(self.n-i))))
            self.Z[i] = np.kron(
                np.eye(2**(i-1)), np.kron(self.sigma_z, np.eye(2**(self.n-i))))

    def generate_photon_operators(self):
        # Generate the photon operators for the photon fock space
        if self.n == 0:
            self.a_ph = 0
            self.a_dag_ph = 0
            self.n_ph = 0
        elif self.n == 1:
            self.a_ph = np.array([[0, 1], [0, 0]])
            self.a_dag_ph = self.a_ph.T
            self.n_ph = self.a_dag_ph @ self.a_ph
        else:
            self.a_ph = np.diag(np.sqrt(np.arange(1, self.n+1)), k=1)
            self.a_dag_ph = self.a_ph.T
            self.n_ph = self.a_dag_ph @ self.a_ph


class CompositeHilbertSpace:
    def __init__(self, hilbert_spaces_dict, flag_operators=True):
        self.hilbert_spaces_dict = hilbert_spaces_dict
        self.composite_basis = self.generate_composite_basis()
        # dict of composite basis
        self.basis_to_index = {base: i for i,
            base in enumerate(self.composite_basis)}
        # without operators, only basis (don't care about system type)
        # check if there is a photon system
        if any([hs.system == "photon" for hs in self.hilbert_spaces_dict.values()]):
            # get index of photon system
            tmp_idx = [i for i, hs in enumerate(
                self.hilbert_spaces_dict.values()) if hs.system == "photon"]
            self.idx_photon = tmp_idx[0] if len(tmp_idx) != 0 else None
            self.flag_photon = True
        else:
            self.flag_photon = False
        # create indices list of spins
        self.idx_spins = [i for i, hs in enumerate(
            self.hilbert_spaces_dict.values()) if hs.system == "spin"]
        if flag_operators:
            self.composite_operators = self.generate_composite_operators()
            # assume one photonic system (cavity) and one+ spin systems, in this order
            # check if the order is correct in the dict, first has to be photon system, then spins
            # assert list(self.hilbert_spaces_dict.values())[0].system != "photon", "First system has to be photon system"
            # assert all([hs.system != "spin" for hs in list(self.hilbert_spaces_dict.values())[1:]]), "Not first systems are not spin systems"
        # flag to store partial composite basis
        self.flag_partial_composite_basis = False

    def generate_composite_basis(self):
        hs_list = list(self.hilbert_spaces_dict.values())
        i = 0
        tmp_basis = hs_list[i].basis
        while (i+1 < len(hs_list)):
            curr_basis = tmp_basis
            tmp_basis = []
            for base_v1 in curr_basis:
                for base_v2 in hs_list[i+1].basis:
                    tmp_basis.append(base_v1+base_v2)
            i += 1
        return tmp_basis

    def generate_composite_operators(self):
        dimensions = [hs.dimension for hs in self.hilbert_spaces_dict.values()]
        self.dimension = np.prod(dimensions)
        # photon operators
        dim_spins = np.prod([dimensions[i] for i in self.idx_spins])
        if self.flag_photon:
            ph_space = list(self.hilbert_spaces_dict.values())[self.idx_photon]
            self.a_ph = np.kron(ph_space.a_ph, np.eye(dim_spins))
            self.a_dag_ph = self.a_ph.T
            self.n_ph = self.a_dag_ph @ self.a_ph
            self.g_2_correlator_operator = self.a_dag_ph @ self.a_dag_ph @ self.a_ph @ self.a_ph
        # spins operators
        self.X = {}
        self.Y = {}
        self.Z = {}
        op_idx = 1
        for i, hs in enumerate(self.hilbert_spaces_dict.values()):
            if hs.system == "spin":
                prev_dim = np.prod(dimensions[:i]) if len(
                    dimensions[:i]) != 0 else 1
                next_dim = np.prod(
                    dimensions[i+1:]) if len(dimensions[i+1:]) != 0 else 1
                # kron prev spaces as id, loop over current op and cron later ones as id too
                for j in range(1, hs.n+1):
                    self.X[op_idx] = np.kron(
                        np.eye(prev_dim), np.kron(hs.X[j], np.eye(next_dim)))
                    self.Y[op_idx] = np.kron(
                        np.eye(prev_dim), np.kron(hs.Y[j], np.eye(next_dim)))
                    self.Z[op_idx] = np.kron(
                        np.eye(prev_dim), np.kron(hs.Z[j], np.eye(next_dim)))
                    op_idx += 1
        # total operators
        n_spins = np.sum(
            [hs.n for hs in self.hilbert_spaces_dict.values() if hs.system == "spin"])
        self.X_total = self.X[1].copy()
        self.Y_total = self.Y[1].copy()
        self.Z_total = self.Z[1].copy()
        for i in range(2, n_spins+1):
            self.X_total += self.X[i]
            self.Y_total += self.Y[i]
            self.Z_total += self.Z[i]
        # raising and lowering operators
        self.S_plus = self.X_total + 1j*self.Y_total
        self.S_minus = self.X_total - 1j*self.Y_total

    def get_reduced_density_matrix(self, rho, keys_to_keep):
        # trace out the keys_to_keep subsystems (they must be consecutive and at the borders of the dict)
        # dimensions of the subsystems
        subsystem_dims = np.prod(
            [self.hilbert_spaces_dict[key].dimension for key in keys_to_keep])
        rho_subsystem = np.zeros(
            (subsystem_dims, subsystem_dims), dtype=complex)
        # get complementary subsystems keys
        keys_to_trace_out = [
            key for key in self.hilbert_spaces_dict.keys() if key not in keys_to_keep]
        # check if subsystems basis is already stored, if yes, use it, otherwise build it
        if self.flag_partial_composite_basis and tuple(keys_to_keep) in self.indices_storage.keys():
            # use stored indices for the trace
            for i in range(subsystem_dims):
                for j in range(subsystem_dims):
                    indices = self.indices_storage[tuple(
                        keys_to_keep)][i, j]
                    sum_tmp = 0
                    for idx in indices:
                        sum_tmp += rho[idx[0], idx[1]]
                    rho_subsystem[i, j] = sum_tmp
            return rho_subsystem
        else:
            # build dict of subsystems spaces
            subsystems_dict = {}
            for key in keys_to_keep:
                subsystems_dict[key] = self.hilbert_spaces_dict[key]
            # build subsystems composite space
            subsystems_composite_space = CompositeHilbertSpace(
                subsystems_dict, flag_operators=False)
            # store subsystems basis
            self.partial_composite_basis = {}
            self.partial_composite_basis[tuple(
                keys_to_keep)] = subsystems_composite_space.composite_basis
            # set flag
            self.flag_partial_composite_basis = True
            # delete subsystems composite space
            del subsystems_composite_space
            # build dict of complementary subsystems spaces
            subsystems_dict = {}
            for key in keys_to_trace_out:
                subsystems_dict[key] = self.hilbert_spaces_dict[key]
            # build complementary subsystems composite space
            subsystems_composite_space = CompositeHilbertSpace(
                subsystems_dict, flag_operators=False)
            # store complementary subsystems basis
            self.partial_composite_basis[tuple(
                keys_to_trace_out)] = subsystems_composite_space.composite_basis
            # delete subsystems composite space
            del subsystems_composite_space
        # check if keys_to_keep are before or after the keys_to_trace_out
        idx_keys_dict = {key: i for i, key in enumerate(
            self.hilbert_spaces_dict.keys())}
        if idx_keys_dict[keys_to_keep[0]] < idx_keys_dict[keys_to_trace_out[0]]:
           to_keep_first = True
        else:
            to_keep_first = False
        # trace out the complementary subsystems
        self.indices_storage = {}
        self.indices_storage[tuple(keys_to_keep)] = {}
        for i, v_tk in enumerate(self.partial_composite_basis[tuple(keys_to_keep)]):
            for j, w_tk in enumerate(self.partial_composite_basis[tuple(keys_to_keep)]):
                sum_tmp = 0
                self.indices_storage[tuple(keys_to_keep)][i, j] = []
                for v_tto in self.partial_composite_basis[tuple(keys_to_trace_out)]:
                    if to_keep_first:
                        v = v_tk + v_tto
                        w = w_tk + v_tto
                    else:
                        v = v_tto + v_tk
                        w = v_tto + w_tk
                    sum_tmp += rho[self.basis_to_index[v],
                        self.basis_to_index[w]]
                    # store indices
                    self.indices_storage[tuple(keys_to_keep)][i, j].append(
                        (self.basis_to_index[v], self.basis_to_index[w]))
                rho_subsystem[i, j] = sum_tmp
        return rho_subsystem


class DensityMatrix:
    def __init__(self, input):
        # check if input is 1d array or 2d array
        if input.ndim == 1:
            self.state = input
            self.state = self.normalize()
            self.dm = self.get_density_matrix()
        elif input.ndim == 2:
            # check if trace of input is 1
            if np.trace(input) != 1:
                self.dm = input / np.trace(input)
            else: 
                self.dm = input
        else:
            raise ValueError("Input must be a 1d or 2d array")

    def normalize(self):
        # Implement the logic to normalize the state
        return self.state / np.linalg.norm(self.state)

    def get_density_matrix(self):
        # Implement the logic to compute the density matrix of the state
        return np.outer(self.state, self.state.conj())

    def get_expectation_value(self, operator):
        # Implement the logic to compute the expectation value of an operator on the state
        tmp = operator @ self.dm
        return np.trace(tmp).real
    def __str__(self):
        return str(self.dm)