document.addEventListener("DOMContentLoaded", function () {
    const seeMoreLink = document.querySelector(".see-more-link");
    // Only rendered when there are more posts than posts_to_show
    if (!seeMoreLink) return;

    const hiddenPosts = document.querySelectorAll(".post.hidden");

    seeMoreLink.addEventListener("click", function (e) {
        e.preventDefault();

        hiddenPosts.forEach(post => {
            post.classList.remove("hidden");
        });

        // Remove the whole <li>, not just the link: the empty item would
        // otherwise keep its cell in the posts grid and leave a gap
        seeMoreLink.closest(".see-more-item").remove();
    });
});
