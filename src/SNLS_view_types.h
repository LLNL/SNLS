#pragma once

#include "SNLS_config.h"
#include "SNLS_gpu_portability.h"

#if defined(SNLS_RAJA_PORT_SUITE) || defined(SNLS_RAJA_ONLY)
#include "RAJA/RAJA.hpp"
#endif

namespace snls
{

#if defined(SNLS_RAJA_PORT_SUITE) || defined(SNLS_RAJA_ONLY)
   // Provide some simple shortcuts in-case people need something beyond the default
   template<typename T>
   using rview1 = RAJA::View<T, RAJA::Layout<1>>;
   template<typename T>
   using rview2 = RAJA::View<T, RAJA::Layout<2>>;
   template<typename T>
   using rview3 = RAJA::View<T, RAJA::Layout<3>>;

   using rview1b = rview1<bool>;
   using rview1d = rview1<double>;
   using rview2d = rview2<double>;
   using rview3d = rview3<double>;
   using crview1d = rview1<const double>;
   using crview2d = rview2<const double>;
   using crview3d = rview3<const double>;

#endif

namespace experimental {
   // This was largely auto-generated when asking google a SFINAE type solution for the layout issue
   template <class T>
   class has_get_layout {
   private:
      template <class U>
      static constexpr auto check(int) -> decltype(std::declval<U>().get_layout(), std::true_type{}) {
         return std::true_type{};
      }

      template <class>
      static constexpr std::false_type check(...) {
         return std::false_type{};
      }

   public:
      static constexpr bool value = decltype(check<T>(0))::value;
   };

   // This was largely auto-generated when asking google a SFINAE type solution for the layout issue
   template <class T>
   class has_get_data {
   private:
      template <class U>
      static constexpr auto check(int) -> decltype(std::declval<U>().get_data(), std::true_type{}) {
         return std::true_type{};
      }

      template <class>
      static constexpr std::false_type check(...) {
         return std::false_type{};
      }

   public:
      static constexpr bool value = decltype(check<T>(0))::value;
   };
}

template<class V>
__snls_hdev__
constexpr
bool
contains_data(const V& view) {
   if constexpr (experimental::has_get_layout<V>::value) {
      return view.get_layout().size_noproj() > 0;
   } else {
      return view.layout.size_noproj() > 0;
   }
}

   // We really don't care what View class we're using as the sub-view just wraps it up
   // and then allows us to take a slice/window of the original view
   // Should probably work out some form of SFINAE to ensure T that we're templated on
   // is an actual View that we can use.
   template<class T>
   class SubView {
   public:
      constexpr SubView() = delete;

      constexpr SubView& operator=(const SubView& other) = default;

      __snls_hdev__
      constexpr SubView(const int index, T& view) : m_view(&view), m_index(index), m_offset(0) {};

      __snls_hdev__
      constexpr SubView(const int index, const size_t offset, T& view) : m_view(&view), m_index(index), m_offset(offset) {};

      ~SubView() = default;

      // Let the compiler figure out the correct return type here as the one from
      // RAJA at least for regular Views is non-trivial
      // make the assumption here that we're using row-major memory order for views
      // so m_index is in the location of the slowest moving index as this is the default
      // for RAJA...
      template <typename Arg0, typename... Args>
      __snls_hdev__
      constexpr
      auto&
      operator()(Arg0 arg0, Args... args) const
      {
         if constexpr (experimental::has_get_layout<T>::value) {
            return (*m_view)(m_index, m_offset + arg0, args...);
         }
         else {
            return (*m_view)(m_offset + arg0, m_index, args...);
         }
      }

      // Needed another operator() overload where we don't supply any arguments as the
      // generic version doesn't work when we are working with 1D m_views and have the
      // equivalent 0D view type
      __snls_hdev__
      constexpr
      auto&
      operator()() const
      {
         return (*m_view)(m_index);
      }

      // If we need to have like a rolling subview/window type class then
      // we'd need some way to update the offset in our slowest moving index
      // in the subview (so not m_view's slowest index)
      __snls_hdev__
      constexpr
      void set_offset(const int offset)
      {
         // Might want an assert in here for debugs to make sure that this is within
         // the bounds of what m_view expects is a valid offset
         m_offset = offset;
      }

      __snls_hdev__
      constexpr
      auto const&
      get_layout() const {
         if constexpr (experimental::has_get_layout<T>::value) {
            return m_view->get_layout();
         } else {
            return m_view->layout;
         }
      }

      // Note if you're using a multi-view then this data type is likely what you expect
      // You should get a pointer but it'll be a D** and not D* where D is the data type of the view.
      __snls_hdev__
      constexpr
      auto*
      get_data() const {
         if constexpr (experimental::has_get_data<T>::value) {
            return m_view->get_data();
         } else {
            return m_view->data;
         }
      }

      T* m_view;
      size_t m_index;
      size_t m_offset;
   };
}
