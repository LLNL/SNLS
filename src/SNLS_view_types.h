#pragma once

#include "SNLS_config.h"
#include "SNLS_gpu_portability.h"

namespace snls
{

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

   // We really don't care what View class we're using as the sub-view just wraps it up
   // and then allows us to take a slice/window of the original view
   // Should probably work out some form of SFINAE to ensure T that we're templated on
   // is an actual View that we can use.
   template<class T>
   class SubView {
   public:

      __snls_hdev__
      SubView() = delete;

      SubView(const SubView& ) = default;

      SubView(SubView&&) = default;

      // SubView& operator=(const SubView&) = default;

      __snls_hdev__
      SubView& operator=(const SubView& other) {
         static_assert(std::is_pointer_v<T>,
         "Subview operator= requires the original view type passed in to be a pointer type and not reference type");
         m_view = other.m_view;
         m_index = other.m_index;
         m_offset = other.m_offset;
         return *this;
      }

      SubView& operator=(SubView&&) = default;

      // Various constructors for when we want a reference type
      template <typename U = T, std::enable_if_t<!std::is_pointer_v<U>,bool> = true>
      __snls_hdev__
      constexpr SubView(const int index, T& view) : m_view(view), m_index(index), m_offset(0) {};

      template <typename U = T, std::enable_if_t<!std::is_pointer_v<U>,bool> = true>
      __snls_hdev__
      constexpr SubView(const int index, const size_t offset, T& view) : m_view(view), m_index(index), m_offset(offset) {};

      // Various constructors for when we want a pointer type
      template <typename U = T, std::enable_if_t<std::is_pointer_v<U>,bool> = true>
      __snls_hdev__
      constexpr SubView(const int index, T view) : m_view(view), m_index(index), m_offset(0) {};

      template <typename U = T, std::enable_if_t<std::is_pointer_v<U>,bool> = true>
      __snls_hdev__
      constexpr SubView(const int index, const size_t offset, T view) : m_view(view), m_index(index), m_offset(offset) {};

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
         if constexpr( std::is_pointer_v<T>) {
               return (*m_view)(m_index, m_offset + arg0, args...);
         }
         else {
               return (m_view)(m_index, m_offset + arg0, args...);
         }
      }

      // Needed another operator() overload where we don't supply any arguments as the
      // above does not work for more complex cases
      template<typename... Args>
      __snls_hdev__
      constexpr
      auto&
      operator()(Args... args) const
      {
         if constexpr( std::is_pointer_v<T>) {
            return (*m_view)(m_index, m_offset + args...);
         }
         else {
            return (m_view)(m_index, m_offset + args...);
         }
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

   private:
      T m_view;
      size_t m_index;
      size_t m_offset;
   };

}