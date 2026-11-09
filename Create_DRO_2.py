# -*- coding: utf-8 -*-


# Create DRO from 2CXM model

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random



def gamma_func(K,alpha,beta,time):

    gamma = (K*(time)**alpha)*np.exp(-(time)/beta)
    return gamma

def phys_params_to_tt_params(f,vp,ve,perm_surf):

    tb = vp/f
    tp = vp/(perm_surf+f)
    te = ve/perm_surf
    return tb,tp,te

def tt_params_to_model_params(tb,tp,te):

    k_m = 0.5*(1.0/tp + 1.0/te - np.sqrt((1.0/tp + 1.0/te)**2 - 4.0*(1.0/tb)*(1.0/te) ))
    k_p = 0.5*(1.0/tp + 1.0/te + np.sqrt((1.0/tp + 1.0/te)**2 - 4.0*(1.0/tb)*(1.0/te) ))
    e = (k_p - 1.0/tb)/(k_p-k_m)
    d = k_p-k_m

    return d,e,k_m

def two_comp_ex_model(parameters,time,AIF):

    [flow,vol_p,vol_e,PS] = parameters
    [Tb,Tp,Te] = phys_params_to_tt_params(flow,vol_p,vol_e,PS)
    [delta,E,km] = tt_params_to_model_params(Tb,Tp,Te)
    return (time[1] - time[0])*flow*(E*np.convolve(np.exp(-km*time),AIF)+(1-E)*np.convolve(np.exp(-((km+delta)*time)),AIF))[:len(AIF)]

def residual(parameters,time,AIF,curve):

    sigm = 1
    r = curve -  two_comp_ex_model(parameters,time,AIF)
    return np.sum(r**2)/(len(curve))




def make_dro_no_delay(tf_format, ps_slice, snr):

    dro_dims = (40, 120, 3)

    # aif
    aif_time = np.around(np.arange(0, 2.0, 0.02), decimals=2).astype(np.float32)
    aif = gamma_func(10 ** 5 / 3.0, 2.0, 0.1, aif_time)

    # set parameter volumes
    flows = np.empty(dro_dims)
    for i, flow in enumerate((0.5, 1.0, 1.5, 2.0)):
        flows[i * 10 : (i + 1) * 10] = flow
    vps = np.empty(dro_dims)
    for i, vp in enumerate((0.02, 0.05, 0.1, 0.2)):
        for j in range(3):
            vps[:, 10 * (i + 4 * j) : 10 * (i + 1 + 4 * j)] = vp
    ves = np.empty(dro_dims)
    for i, ve in enumerate((0.1, 0.2, 0.5)):
        ves[:, i * 40 : (i + 1) * 40] = ve
    pss = np.empty(dro_dims)
    for i, ps in enumerate((0.5, 1.5, 2.5)):
        pss[..., i] = ps

    myo_pars = np.stack((flows, vps, ves, pss), axis=-1)[..., np.newaxis, :].astype(
        np.float32
    )
    myo_curves = np.empty((len(aif),) + dro_dims + (1,)).astype(np.float32)
    myo_t_curves = np.empty_like(myo_curves)

    for i in range(dro_dims[0]):
        for j in range(dro_dims[1]):
            for k in range(dro_dims[2]):
                signal = two_comp_ex_model(
                    myo_pars[i, j, k, 0, :], aif_time, aif
                )

                # v approximates v_e + v_p
                myo_curves[:, i, j, k, 0] = signal
                myo_t_curves[:, i, j, k, 0] = np.gradient(
                    myo_curves[:, i, j, k, 0], aif_time[1] - aif_time[0]
                )

    if snr:
        # add noise
        signal_db_mean = 10 * np.log10(np.mean(myo_curves))
        noise_db_mean = signal_db_mean - snr
        noise_pw_mean = 10 ** (noise_db_mean / 10)
        noise_pw = np.random.normal(0.0, np.sqrt(noise_pw_mean), myo_curves.shape)
        myo_curves += noise_pw

    # remove impossible values
    myo_curves[0, ...] = 0
    # myo_curves = np.maximum(myo_curves, 0)

    # input normalization
    # Commented to exclude negative aif times
    # t_std = aif_time.std()
    # aif_time = (aif_time - aif_time.mean()) / aif_time.std()

    # output normalization
    max_ = aif.max()
    aif /= max_
    myo_curves /= max_
    myo_t_curves /= max_

    # r and b
    coll_points = np.random.uniform(
        min(aif_time), max(aif_time), len(aif_time) * 5
    ).astype(np.float32)
    bound = np.array([min(aif_time)])
    extrapolate = np.random.uniform(
        max(aif_time), max(aif_time) * 3, len(aif_time) * 5
    ).astype(np.float32)

    # select slice
    myo_pars = myo_pars[..., ps_slice, :, :]
    myo_curves = myo_curves[..., ps_slice, :]
    myo_t_curves = myo_t_curves[..., ps_slice, :]

    if tf_format == -1:
        # return as numpy arrays
        return aif_time, aif, myo_curves, myo_pars
    else:
        raise ValueError("Wrong input format, please type -1 for numpy arrays")





def make_dro_delay(tf_format, ps_slice, snr, maximum_delay=3):
    dro_dims = (40, 120, 3)

    # aif
    aif_time = np.around(np.arange(0, 2.0, 0.02), decimals=2).astype(np.float32)
    aif = gamma_func(10 ** 5 / 3.0, 2.0, 0.1, aif_time)

    # set parameter volumes
    flows = np.empty(dro_dims)
    for i, flow in enumerate((0.5, 1.0, 1.5, 2.0)):
        flows[i * 10 : (i + 1) * 10] = flow
    vps = np.empty(dro_dims)
    for i, vp in enumerate((0.02, 0.05, 0.1, 0.2)):
        for j in range(3):
            vps[:, 10 * (i + 4 * j) : 10 * (i + 1 + 4 * j)] = vp
    ves = np.empty(dro_dims)
    for i, ve in enumerate((0.1, 0.2, 0.5)):
        ves[:, i * 40 : (i + 1) * 40] = ve
    pss = np.empty(dro_dims)
    for i, ps in enumerate((0.5, 1.5, 2.5)):
        pss[..., i] = ps
    
    
    myo_curves = np.empty((len(aif),) + dro_dims + (1,)).astype(np.float32)
    myo_t_curves = np.empty_like(myo_curves)

    block_size = 10
    delay_map = np.empty(dro_dims, dtype=np.float32)
    for i in range(0, dro_dims[0], block_size):
        for j in range(0, dro_dims[1], block_size):
            d = random.randint(0, maximum_delay)
            delay_map[i:i+block_size, j:j+block_size, :] = d
    
    myo_pars = np.stack((flows, vps, ves, pss, delay_map), axis=-1)[..., np.newaxis, :].astype(
        np.float32
    )
    for i in range(dro_dims[0]):
        for j in range(dro_dims[1]):
            for k in range(dro_dims[2]):
                signal = two_comp_ex_model(
                    myo_pars[i, j, k, 0, :-1], aif_time, aif
                )
                d = int(myo_pars[i, j, k, 0,-1])  # repeat same delay for all k
                if d > 0:
                    myo_curves[d:, i, j, k, 0] = signal[:-d]
                    myo_curves[:d, i, j, k, 0] = 0
                else:
                    myo_curves[:, i, j, k, 0] = signal
                # v approximates v_e + v_p
                #myo_curves[:, i, j, k, 0] = signal 
                # myo_t_curves[:, i, j, k, 0] = np.gradient(ø
                #     myo_curves[:, i, j, k, 0], aif_time[1] - aif_time[0]
                # )
     
   

    if snr:
        # add noise
        signal_db_mean = 10 * np.log10(np.mean(myo_curves))
        noise_db_mean = signal_db_mean - snr
        noise_pw_mean = 10 ** (noise_db_mean / 10)
        noise_pw = np.random.normal(0.0, np.sqrt(noise_pw_mean), myo_curves.shape)
        myo_curves += noise_pw

    # remove impossible values
    myo_curves[0, ...] = 0
    # myo_curves = np.maximum(myo_curves, 0)

    # input normalization
    # Commented to exclude negative aif times
    # t_std = aif_time.std()
    # aif_time = (aif_time - aif_time.mean()) / aif_time.std()

    # output normalization
    max_ = aif.max()
    aif /= max_
    myo_curves /= max_
    myo_t_curves /= max_

    # r and b
    coll_points = np.random.uniform(
        min(aif_time), max(aif_time), len(aif_time) * 5
    ).astype(np.float32)
    bound = np.array([min(aif_time)])
    extrapolate = np.random.uniform(
        max(aif_time), max(aif_time) * 3, len(aif_time) * 5
    ).astype(np.float32)

    # select slice
    myo_pars = myo_pars[..., ps_slice, :, :]
    myo_curves = myo_curves[..., ps_slice, :]
    myo_t_curves = myo_t_curves[..., ps_slice, :]

    if tf_format == -1:
        # return as numpy arrays
        return aif_time, aif, myo_curves, myo_pars
    else:
        print("Wrong input format, please type -1 for numpy arrays")







if __name__ == "__main__":
    
    aif_time, aif, myo_curves, myo_pars = make_dro_no_delay(-1, 0, 17.5) # type: ignore

    print(myo_curves.shape)
    print(myo_pars.shape)
    #Debug plot
    plt.imshow(myo_curves[15,..., 0].T, cmap='grey')
    plt.savefig("aaaaaaaaa.png")
    
    plt.show()
    plt.close()

    fig, axs = plt.subplots(1, 5, figsize=(20, 5))
    im0 = axs[0].imshow(myo_pars[...,0,0].T, cmap='plasma')
    im1 = axs[1].imshow(myo_pars[...,0,1].T, cmap='plasma')
    im2 = axs[2].imshow(myo_pars[...,0,2].T, cmap='plasma')
    im3 = axs[3].imshow(myo_pars[...,0,3].T, cmap='plasma')
    #im4 = axs[4].imshow(myo_pars[...,0,4].T, cmap='plasma')
    axs[0].set_title("Fp")
    axs[1].set_title("vp")
    axs[2].set_title("ve")
    axs[3].set_title("PS")
    #axs[4].set_title("Delay")

    for ax in axs:
        ax.axis("off")


    cbar0 = fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)
    #cbar0.set_label(" [ml]")

    cbar1 = fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)
    #cbar1.set_label("[m/s]")

    cbar2 = fig.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)
    #cbar2.set_label("[m/s]")

    cbar3 = fig.colorbar(im3, ax=axs[3], fraction=0.046, pad=0.04)
    #cbar3.set_label("")

    #cbar4 = fig.colorbar(im4, ax=axs[4], fraction=0.046, pad=0.04)
    #cbar4.set_label("Delay [s]")

    plt.tight_layout()
    plt.savefig("parameter_map_dro2_with_individual_legends.png")
    plt.show()
    plt.close()


    params_scatter = myo_pars.squeeze(axis=2)  
    flat_params = params_scatter.reshape(-1, 5)
    Fp_vals  = flat_params[:, 0]  
    vp_vals  = flat_params[:, 1]
    ve_vals  = flat_params[:, 2]
    ps_vals  = flat_params[:, 3]
    delay = flat_params[:, 4]

    fig, axs = plt.subplots(1, 5, figsize=(20, 4))
    axs[0].scatter(range(len(Fp_vals)), Fp_vals, s=3, color='red')
    axs[0].set_title("Fp")
    axs[1].scatter(range(len(vp_vals)), vp_vals, s=3, color='green')
    axs[1].set_title("vp")
    axs[2].scatter(range(len(ve_vals)), ve_vals, s=3, color='blue')
    axs[2].set_title("ve")
    axs[3].scatter(range(len(ps_vals)), ps_vals, s=3, color='purple')
    axs[3].set_title("PS")
    axs[4].scatter(range(len(delay)), delay, s=3, color='orange')
    axs[4].set_title("Delay")
    plt.tight_layout()
    plt.savefig("Scatter_parameters.png")
    plt.show()
    plt.close()


# #     # print(aif.shape)
#     # print(myo_curves.shape)
#     #print(myo_curves[...,20,80,0])
    # print(aif.shape)
    #print(aif_time)
    #plt.plot(aif_time, aif, 'r-.')
    # for i in range(100):
        
    #   plt.imshow(myo_curves[i,...,0], cmap='plasma')
    #   #plt.plot(aif_time, myo_curves[...,30,40,0], 'r-.')
    #   plt.show()
 