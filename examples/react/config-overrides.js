module.exports = {
    // The Webpack config to use when compiling your react app for development or production.
    webpack: function (config, env) {
      // set resolve.fallback
      config.resolve.fallback = {
        fs: false,
        path: false,
        crypto: false,
      };
      return config;
    },
};