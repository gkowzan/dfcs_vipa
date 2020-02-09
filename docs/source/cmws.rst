.. currentmodule:: pydfcs.cmws

pydfcs.cmws
=============

.. automodule:: pydfcs.cmws

   .. rubric:: Comb-cavity lineshape functions

   .. autosummary::
      :toctree: _autosummary/

      load_comb_psd
      calc_comb_lineshape


   .. rubric:: HDF5 conversion functions

   .. autosummary::
      :toctree: _autosummary/

      dir_hdf5old2new
      hdf5old2new
      hdf5old2new_copy
      dir_hdf5old2new_copy


   .. rubric:: Fitting and work-flow functions

   .. autosummary::
      :toctree: _autosummary/

      make_teeth_grid
      get_rio_pos_file
      make_grid_file


   .. rubric:: Frequency axis functions

   .. autosummary::
      :toctree: _autosummary/

      tooth_number


   .. rubric:: Fitting functions

   .. autosummary::
      :toctree: _autosummary/

      lorentz
      lorentz_conv
      lorentz_diff
      voigt
      fit_mode
      fit_modes


   .. rubric:: Fitting functions

   .. autosummary::
      :toctree: _autosummary/

      init_calibrate
      worker_calibrate
      calibrate_collection_mp
      calibrate_collection
      make_cal_func
      splev_wrap


   .. rubric:: Data persistence functions

   .. autosummary::
      :toctree: _autosummary/

      save_h5
