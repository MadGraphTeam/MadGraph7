#pragma once

#include <cstdio>
#include <format>
#include <ranges>
#include <tuple>
#include <vector>

namespace madspace {

template <class... Ts>
struct Overloaded : Ts... {
    using Ts::operator()...;
};
template <class... Ts>
Overloaded(Ts...) -> Overloaded<Ts...>;

template <typename T>
using nested_vector2 = std::vector<std::vector<T>>;
template <typename T>
using nested_vector3 = std::vector<std::vector<std::vector<T>>>;
template <typename T>
using nested_vector4 = std::vector<std::vector<std::vector<std::vector<T>>>>;

// Unfortunately nvcc does not support C++23 yet, so we implement our own zip function
// here (based on https://github.com/alemuntoni/zip-views), otherwise use the standard
// library function

namespace detail {

inline void print_impl(
    std::FILE* stream, bool new_line, std::string_view fmt, std::format_args args
) {
    std::string str = std::vformat(fmt, args);
    if (new_line) {
        str.push_back('\n');
    }
    fwrite(str.data(), 1, str.size(), stream);
}

template <typename... Args, std::size_t... Index>
bool any_match_impl(
    const std::tuple<Args...>& lhs,
    const std::tuple<Args...>& rhs,
    std::index_sequence<Index...>
) {
    auto result = false;
    result = (... || (std::get<Index>(lhs) == std::get<Index>(rhs)));
    return result;
}

template <typename... Args>
bool any_match(const std::tuple<Args...>& lhs, const std::tuple<Args...>& rhs) {
    return any_match_impl(lhs, rhs, std::index_sequence_for<Args...>{});
}

template <std::ranges::viewable_range... Rng>
class zip_iterator {
public:
    using value_type = std::tuple<std::ranges::range_reference_t<Rng>...>;

    zip_iterator() = delete;
    zip_iterator(std::ranges::iterator_t<Rng>&&... iters) :
        _iters{std::forward<std::ranges::iterator_t<Rng>>(iters)...} {}

    zip_iterator& operator++() {
        std::apply([](auto&&... args) { ((++args), ...); }, _iters);
        return *this;
    }

    zip_iterator operator++(int) {
        auto tmp = *this;
        ++*this;
        return tmp;
    }

    bool operator!=(const zip_iterator& other) const { return !(*this == other); }

    bool operator==(const zip_iterator& other) const {
        return any_match(_iters, other._iters);
    }

    value_type operator*() {
        return std::apply([](auto&&... args) { return value_type(*args...); }, _iters);
    }

private:
    std::tuple<std::ranges::iterator_t<Rng>...> _iters;
};

template <std::ranges::viewable_range... T>
class zipper {
public:
    using zip_type = zip_iterator<T...>;

    template <typename... Args>
    zipper(Args&&... args) : _args{std::forward<Args>(args)...} {}

    zip_type begin() {
        return std::apply(
            [](auto&&... args) { return zip_type(std::ranges::begin(args)...); }, _args
        );
    }
    zip_type end() {
        return std::apply(
            [](auto&&... args) { return zip_type(std::ranges::end(args)...); }, _args
        );
    }

private:
    std::tuple<T...> _args;
};

} // namespace detail

template <std::ranges::viewable_range... T>
auto zip(T&&... t) {
    return detail::zipper<T...>{std::forward<T>(t)...};
}

template <typename... Args>
inline void print(std::format_string<Args...> fmt, Args&&... args) {
    detail::print_impl(stdout, false, fmt.get(), std::make_format_args(args...));
}

template <typename... Args>
inline void print(std::FILE* stream, std::format_string<Args...> fmt, Args&&... args) {
    detail::print_impl(stream, false, fmt.get(), std::make_format_args(args...));
}

template <typename... Args>
inline void println(std::format_string<Args...> fmt, Args&&... args) {
    detail::print_impl(stdout, true, fmt.get(), std::make_format_args(args...));
}

template <typename... Args>
inline void
println(std::FILE* stream, std::format_string<Args...> fmt, Args&&... args) {
    detail::print_impl(stream, true, fmt.get(), std::make_format_args(args...));
}

} // namespace madspace
