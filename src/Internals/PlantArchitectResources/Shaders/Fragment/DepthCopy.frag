layout (location = 0) out vec3 fragColor;

in VS_OUT {
    vec2 TexCoords;
} fs_in;

uniform sampler2D depthStencil;
uniform float near;
uniform float far;
uniform float factor;
float CalculateLinearDepth(float ndcDepth, float near, float far)
{
    float z = ndcDepth * 2.0 - 1.0;
    return (2.0 * near * far) / (far + near - z * (far - near));
}

void main()
{
    float ndcDepth = texture(depthStencil, fs_in.TexCoords).r;
    float depth = CalculateLinearDepth(ndcDepth, near, far) / factor;

    fragColor = vec3(depth, depth, depth);
}