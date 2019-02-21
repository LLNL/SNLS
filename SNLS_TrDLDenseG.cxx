#include "SNLS_cuda_portability.h"
#include "SNLS_TrDLDenseG.h"

namespace snls {

// constructor
__snls_hdev__
TrDeltaControl::TrDeltaControl() :
   _xiLG(0.75),
   _xiUG(1.4),
   _xiIncDelta(1.5),
   _xiLO(0.35),
   _xiUO(5.0),
   _xiDecDelta(0.25),
   _xiForcedIncDelta(1.2),
   _deltaInit(1.0),
   _deltaMin(1e-12),
   _deltaMax(1e4),
   _rejectResIncrease(true)
{
   this->checkParams() ;
}

__snls_hdev__
void TrDeltaControl::checkParams() const
{
   if ( ( _deltaMin <= 0 ) ||
        ( _deltaMax <= _deltaMin ) ||
        ( _xiLG <= _xiLO ) ||
        ( _xiUG >= _xiUO ) ||
        ( _xiIncDelta <= 1 ) ||
        ( ( _xiDecDelta >= 1 ) || ( _xiDecDelta <= 0 ) ) ||
        ( _xiForcedIncDelta <= 1 ) 
        ) {
      SNLS_FAIL("TrDeltaControl::checkParams", "bad setting") ;
   }

}

__snls_hdev__   
bool TrDeltaControl::decrDelta(void *strm, real8 &delta, real8 normfull, bool took_full) const
{
   bool success = true ;
   
   if ( took_full ) 
   {
      real8 tempa = delta    * _xiDecDelta ;
      real8 tempb = normfull * _xiDecDelta ;
      delta = sqrt( tempa*tempb ) ;
   }
   else 
       delta = delta * _xiDecDelta ;

   if ( delta < _deltaMin ) 
   {
      delta = _deltaMin ;

#ifdef __cuda_host_only__
      if (strm) { *((std::ostream *) strm) << "delta now at min " << delta << std::endl ; }
#endif

      success = false ;
   }

#ifdef __cuda_host_only__
   else 
      if (strm) { *((std::ostream *) strm) << "decr delta to " << delta << std::endl ; }
#endif

   return success;
}

__snls_hdev__
void TrDeltaControl::incrDelta(void* strm, real8  &delta) const
{
   delta = delta * _xiIncDelta;

   if ( delta > _deltaMax ) 
   {
      delta = _deltaMax ;

#ifdef __cuda_host_only__
      if (strm) { *((std::ostream *) strm) << "delta now at max " << delta << std::endl ; }
#endif
   }
#ifdef __cuda_host_only__
   else 
      if (strm) { *((std::ostream *) strm) << "incr delta to "    << delta << std::endl ; }
#endif
}

__snls_hdev__
bool TrDeltaControl::updateDelta
(
   void   *strm      ,
   real8  &delta     ,
   real8   res       ,
   real8   res_0     ,
   real8   pred_resid,
   bool   &reject    , // reject_prev
   bool    took_full , // use_nr
   real8   normfull  , // nr2norm
   real8  &rho       ) const
{
   bool  success       = true;
   real8 actual_change = res - res_0;
   real8 pred_change   = pred_resid - res_0;

   if ( pred_change == 0e0 ) 
   {
      if ( delta >= _deltaMax ) {
         // things are going badly enough that the solver should probably stop
#ifdef __cuda_host_only__
         if (strm) { *((std::ostream *) strm) << "predicted change is zero and delta at max" << std::endl; }
#endif
         success = false ;
      }
      else {
#ifdef __cuda_host_only__
         if (strm) { *((std::ostream *) strm) << "predicted change is zero, forcing delta larger" << std::endl; }
#endif
         delta = fmin( delta * _xiForcedIncDelta, _deltaMax );
         // ierr = IERR_UPD_DELTA_PRZ_p; // do not report this as a failure
      }
   }
   else 
   {
      rho = actual_change / pred_change;
#ifdef __cuda_host_only__
      if (strm) { *((std::ostream *) strm) << "rho = " << rho << std::endl; }
#endif
      if ( ( rho > _xiLG ) &&
           ( actual_change < 0e0 ) &&
           ( rho < _xiUG ) ) {
         if ( ! took_full ) {
            // increase delta
            this->incrDelta(strm, delta);
         }
         // instead of truncating, just do not increase, as 
         // truncating can cause trouble
         // else { 
         // 
         //    real8 temp = delta;
         //    real8 temp_b = normfull * _xiIncDelta;
         //    if (temp_b < delta) {
         //       delta = temp_b;
#ifdef __cuda_host_only__
         //       if ( _os != NULL ) {
         //          _os << "took full step, delta truncated from " << temp << " to " << delta << std::endl;
         //       }
#endif
         //    }
         // }
      }
      else if ( ( rho < _xiLO ) ||
                ( rho > _xiUO ) ) 
      { 
         // decrease delta
         success = this->decrDelta(strm, delta, normfull, took_full);
      }
   } // pred_change == 0
   
   reject = false;

   // do not make this >= , may have res and res_0 both zero and that is ok
   if ( ( actual_change > 0e0 ) && 
        ( _rejectResIncrease ) ) 
   { 
      // residual increasing, need to reject solution
      // have already decreased delta (rho < 0)
      reject = true;
   }
   
   return success;
}

} // namespace snls
